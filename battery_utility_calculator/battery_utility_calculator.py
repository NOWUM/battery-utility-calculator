# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

from typing import Literal

import pandas as pd

from battery_utility_calculator import EnergyCostCalculator as ECC
from battery_utility_calculator import Storage


def calculate_storage_worth(
    baseline_storage: Storage,
    storage_to_calculate: Storage,
    demand: pd.Series,
    solar_generation: pd.Series,
    grid_prices: pd.Series,
    eeg_prices: pd.Series,
    community_market_prices: pd.Series,
    wholesale_market_prices: pd.Series,
    storage_use_cases: list[str] = ["eeg", "home", "community", "wholesale"],
    allow_community_to_home: bool = False,
    allow_community_to_storage: bool = False,
    allow_pv_to_community: bool = False,
    allow_storage_to_wholesale: bool = False,
    solver: str = "gurobi",
) -> float:
    """Calculates the worth (value) of a single storage (compared to a baseline storage).

    Args:
        baseline_storage (Storage): The baseline storage to compare to.
        storage_to_calculate (Storage): The storage to calculate the worth (value) for.
        demand (pd.Series): Demand timeseries. Values should be in kWh per hour (kW).
        solar_generation (pd.Series): Solar generation timeseries. Values should be in kWh per hour (kW).
        grid_prices (pd.Series): Grid prices timeseries. Values should be in EUR per kWh.
        eeg_prices (pd.Series): EEG prices timeseries. Values should be in EUR per kWh.
        community_market_prices (pd.Series): Community market timeseries. Values should be in EUR per kWh.
        wholesale_market_prices (pd.Series): Wholesale market timeseries. ^alues should be in EUR per kWh.
        storage_use_cases (list[str], optional): Use cases for storage. Defaults to ["eeg", "home", "community", "wholesale"].
        allow_community_to_home (bool, optional): Wether to allow using energy from community for home use. Defaults to False.
        allow_community_to_storage (bool, optional): Wether to allow storing energy from community for home use. Defaults to False.
        allow_pv_to_community (bool, optional): Wether to allow selling PV energy to community. Defaults to False.
        allow_storage_to_wholesale (bool, optional): Wether to allow selling from storage to wholesale market. Defaults to False.
        check_timeseries (bool, optional): Wether to check time series. Defaults to True.
        solver (str, optional): Which solver to use. Defaults to "gurobi".

    Returns:
        float: Worth (value) of the storage.
    """

    # calculate baseline costs
    baseline_ecc = ECC(
        storage=baseline_storage,
        demand=demand,
        solar_generation=solar_generation,
        grid_prices=grid_prices,
        eeg_prices=eeg_prices,
        community_market_prices=community_market_prices,
        wholesale_market_prices=wholesale_market_prices,
        storage_use_cases=storage_use_cases,
        allow_community_to_home=allow_community_to_home,
        allow_community_to_storage=allow_community_to_storage,
        allow_pv_to_community=allow_pv_to_community,
        allow_storage_to_wholesale=allow_storage_to_wholesale,
    )
    baseline_costs = baseline_ecc.optimize(solver=solver)

    # calculate costs for storage to calculate
    to_calc_ecc = ECC(
        storage=storage_to_calculate,
        demand=demand,
        solar_generation=solar_generation,
        grid_prices=grid_prices,
        eeg_prices=eeg_prices,
        community_market_prices=community_market_prices,
        wholesale_market_prices=wholesale_market_prices,
        storage_use_cases=storage_use_cases,
        allow_community_to_home=allow_community_to_home,
        allow_community_to_storage=allow_community_to_storage,
        allow_pv_to_community=allow_pv_to_community,
        allow_storage_to_wholesale=allow_storage_to_wholesale,
    )
    to_calc_costs = to_calc_ecc.optimize(solver=solver)

    # storage worth is difference between baseline and new storage
    storage_worth = to_calc_costs - baseline_costs

    return storage_worth


def calculate_multiple_storage_worth(
    baseline_storage: Storage,
    storages_to_calculate: list[Storage],
    demand: pd.Series,
    solar_generation: pd.Series,
    grid_prices: pd.Series,
    eeg_prices: pd.Series,
    community_market_prices: pd.Series,
    wholesale_market_prices: pd.Series,
    solver: str = "gurobi",
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Calculates the worth (value) of multiple storages compared to a baseline storage.

    Args:
        baseline_storage (Storage): The baseline storage to compare to.
        storages_to_calculate (list[Storage]): List of storages to calculate worth (value) for.
        demand (pd.Series): Demand timeseries. Values should be in kWh per hour (kW).
        solar_generation (pd.Series): Solar generation timeseries. Values should be in kWh per hour (kW).
        grid_prices (pd.Series): Grid prices timeseries. Values should be in EUR per kWh.
        eeg_prices (pd.Series): EEG prices timeseries. Values should be in EUR per kWh.
        community_market_prices (pd.Series): Community market timeseries. Values should be in EUR per kWh.
        wholesale_market_prices (pd.Series): Wholesale market timeseries. Values should be in EUR per kWh.
        storage_use_cases (list[str], optional): Use cases for storage. Defaults to ["eeg", "home", "community", "wholesale"].
        allow_community_to_home (bool, optional): Wether to allow using energy from community for home use. Defaults to False.
        allow_community_to_storage (bool, optional): Wether to allow storing energy from community for home use. Defaults to False.
        allow_pv_to_community (bool, optional): Wether to allow selling PV energy to community. Defaults to False.
        allow_storage_to_wholesale (bool, optional): Wether to allow selling from storage to wholesale market. Defaults to False.
        check_timeseries (bool, optional): Wether to check time series. Defaults to True.
        solver (str, optional): Which solver to use. Defaults to "gurobi".

    Returns:
        pd.DataFrame: DataFrame containing storage parameters and their worth (value).
    """

    # calculate baseline costs
    baseline_ecc = ECC(
        storage=baseline_storage,
        demand=demand,
        solar_generation=solar_generation,
        grid_prices=grid_prices,
        eeg_prices=eeg_prices,
        community_market_prices=community_market_prices,
        wholesale_market_prices=wholesale_market_prices,
        *args,
        **kwargs,
    )
    baseline_costs = baseline_ecc.optimize(solver=solver)

    df = pd.DataFrame(
        columns=[
            "id",
            "c_rate",
            "volume",
            "charge_efficiency",
            "discharge_efficiency",
            "costs",
            "worth",
        ]
    )
    df.loc[
        0,
        [
            "id",
            "c_rate",
            "volume",
            "charge_efficiency",
            "discharge_efficiency",
            "costs",
            "worth",
        ],
    ] = [
        baseline_storage.id,
        baseline_storage.c_rate,
        baseline_storage.volume,
        baseline_storage.charge_efficiency,
        baseline_storage.discharge_efficiency,
        baseline_costs,
        0,
    ]

    for storage in storages_to_calculate:
        ecc = ECC(
            storage=storage,
            demand=demand,
            solar_generation=solar_generation,
            grid_prices=grid_prices,
            eeg_prices=eeg_prices,
            community_market_prices=community_market_prices,
            wholesale_market_prices=wholesale_market_prices,
            *args,
            **kwargs,
        )
        costs = ecc.optimize(solver=solver)
        storage_worth = costs - baseline_costs

        stor_df = pd.DataFrame()
        stor_df["id"] = [storage.id]
        stor_df["c_rate"] = [storage.c_rate]
        stor_df["volume"] = [storage.volume]
        stor_df["charge_efficiency"] = [storage.charge_efficiency]
        stor_df["discharge_efficiency"] = [storage.discharge_efficiency]
        stor_df["costs"] = [costs]
        stor_df["worth"] = [storage_worth]

        df = pd.concat([df, stor_df], ignore_index=True)

    return df


def calculate_bidding_curve(
    volumes_worth: pd.DataFrame,
    buy_or_sell_side: Literal["buyer", "seller"],
) -> pd.DataFrame:
    """Calculates the bidding curve for a single product.

    Args:
        volumes_worth (pd.DataFrame): The volumes and their worth (values) in a pd.DataFrame. Columns should be "volume" and "worth".
        buy_or_sell_side (Literal["buyer", "seller"]): Wether to calculate for buyer side or seller side.

    Returns:
        pd.DataFrame: A new DataFrame with the bidding curve.
    """

    if 0 not in volumes_worth["volume"].values:
        volumes_worth.loc[len(volumes_worth), ["volume", "worth"]] = 0, 0

    if buy_or_sell_side == "buyer":
        df = volumes_worth.sort_values("volume", ascending=True)
    elif buy_or_sell_side == "seller":
        df = volumes_worth.sort_values("volume", ascending=True)
    else:
        raise ValueError("buy_or_sell_side has to be either 'buyer' or 'seller'")

    original_costs = df["costs"].copy()

    df = df.diff().dropna().reset_index(drop=True).abs()
    df["costs"] = original_costs
    df["cumulative_volume"] = df["volume"].cumsum()
    df.rename(columns={"worth": "marginal_price"}, inplace=True)
    df = df[df["volume"] != 0].reset_index(drop=True)
    df["marginal_price_per_kwh"] = df["marginal_price"] / df["volume"]
    return df
