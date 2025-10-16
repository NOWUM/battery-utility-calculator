# SPDX-FileCopyrightText: Christoph Komanns, Florian Maurer, Ralf Schemm
#
# SPDX-License-Identifier: MIT

import pandas as pd

from battery_utility_calculator.energy_costs_calculator import (
    EnergyCostCalculator,
    Storage,
)


def test_ECC_baseline():
    # buying 1 kWh for 1 €/kWh should equal to 3€ total
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0, efficiency=1),
        eeg_prices=pd.Series([0, 0, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([1, 1, 1]),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([1, 1, 1]),
    )
    costs = calculator.optimize(solver="highs")
    assert costs == -3


def test_ECC_opti_storage():
    # buying 2 kWh for 0€/kWh and storing 1 kWh of this should equal 1€ total
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
        eeg_prices=pd.Series([0, 0, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([0, 1, 1]),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([1, 1, 1]),
    )
    costs = calculator.optimize(solver="highs")
    assert costs == -1


def test_ECC_opti_storage_2():
    # now we need 2 kWh at each timestep
    # on timestep=0, we can buy for 0€/kWh and should buy 3kWh
    # as we use 2 kWh during timestep=0 and use 1 kWh for timestep=1
    # total cost should be 3*0 + 1*1 + 2*1 = 3
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
        eeg_prices=pd.Series([0, 0, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([0, 1, 1]),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([2, 2, 2]),
    )
    costs = calculator.optimize(solver="highs")
    assert costs == -3


def test_ECC_selling_pv():
    # here we should gain 1€ from selling pv
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=0, efficiency=1),
        eeg_prices=pd.Series([1, 0, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([1, 1, 1]),
        solar_generation=pd.Series([1, 0, 0]),
        demand=pd.Series([0, 0, 0]),
    )
    costs = calculator.optimize(solver="highs")
    assert costs == 1


def test_ECC_selling_pv_w_storage():
    # same as above, but we can store PV and sell at
    # timestep=1 instead of timestep=0, as we can get 2€/kWh
    # in timestep=1
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
        eeg_prices=pd.Series([1, 2, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([1, 1, 1]),
        solar_generation=pd.Series([1, 0, 0]),
        demand=pd.Series([0, 0, 0]),
    )
    costs = calculator.optimize(solver="highs")
    assert costs == 2

    # charge from solar_generation in ts=0,1 and discharge at ts=2
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=2, efficiency=1),
        eeg_prices=pd.Series([0, 0, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([5, 10, 20]),
        solar_generation=pd.Series([1, 1, 0]),
        demand=pd.Series([0, 0, 2]),
    )
    costs = calculator.optimize(solver="highs")
    assert costs == 0


def test_ECC_negative_prices():
    # buy as much in ts=2 cause we get paid for this
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=2, efficiency=1),
        eeg_prices=pd.Series([0, 0, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([5, 10, -20]),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([0, 0, 2]),
    )
    costs = calculator.optimize(solver="highs")
    assert costs == 80


def test_ECC__c_rate():
    # check if c_rate is respected
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=0.5, volume=2, efficiency=1),
        eeg_prices=pd.Series([0, 0, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([0, 10, 0]),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([2, 2, 0]),
    )
    costs = calculator.optimize(solver="highs")
    assert costs == -10
