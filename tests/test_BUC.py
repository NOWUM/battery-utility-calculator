# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

import pandas as pd

from battery_utility_calculator.battery_utility_calculator import (
    Storage,
    calculate_bidding_curve,
    calculate_multiple_storage_worth,
    calculate_storage_worth,
)


def test_calculate_storage_worth():
    baseline_storage = Storage(0, 1, 0, 1)
    storage_to_calc = Storage(0, 1, 1, 1)

    worth = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([0, 1, 1]),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([1, 1, 1]),
        solver="appsi_highs",
    )

    # with baseline storage costs should be 2
    # with storage costs should be 1
    assert worth == 1


def test_calculate_multiple_storage_worth():
    baseline_storage = Storage(0, 1, 0, 1)
    storages_to_calc = [Storage(0, 1, 1, 1), Storage(0, 1, 2, 1)]

    worths = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0]),
        wholesale_market_prices=pd.Series([0, 0, 0]),
        community_market_prices=pd.Series([0, 0, 0]),
        grid_prices=pd.Series([0, 1, 1]),
        solar_generation=pd.Series([0, 0, 0]),
        demand=pd.Series([1, 1, 1]),
        solver="appsi_highs",
    )

    assert (worths["costs"].values == [-2, -1, 0]).all()
    assert pd.isna(worths["worth"][0])
    assert (worths["worth"].values[1:] == [1, 2]).all()


def test_calculate_bidding_curve():
    volume_worths = pd.DataFrame()
    volume_worths["volume"] = [1, 2, 3]
    volume_worths["worth"] = [5, 7, 8]

    bidding_curve = calculate_bidding_curve(
        volumes_worth=volume_worths, buy_or_sell_side="buyer"
    )

    correct_df = pd.DataFrame()
    correct_df["volume"] = [1, 1, 1]
    correct_df["marginal_price"] = [5, 2, 1]

    assert (bidding_curve == correct_df).all().all()
