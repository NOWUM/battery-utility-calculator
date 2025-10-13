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
    check_timeseries: bool = True,
    solver: str = "gurobi",
) -> float:
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
        check_timeseries=check_timeseries,
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
        check_timeseries=check_timeseries,
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
    storage_use_cases: list[str] = ["eeg", "home", "community", "wholesale"],
    allow_community_to_home: bool = False,
    allow_community_to_storage: bool = False,
    allow_pv_to_community: bool = False,
    allow_storage_to_wholesale: bool = False,
    check_timeseries: bool = True,
    solver: str = "gurobi",
) -> pd.DataFrame:
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
        check_timeseries=check_timeseries,
    )
    baseline_costs = baseline_ecc.optimize(solver=solver)

    df = pd.DataFrame(
        columns=["id", "c_rate", "volume", "efficiency", "costs", "worth"]
    )
    for storage in storages_to_calculate:
        ecc = ECC(
            storage=storage,
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
            check_timeseries=check_timeseries,
        )
        costs = ecc.optimize(solver=solver)
        storage_worth = costs - baseline_costs

        stor_df = pd.DataFrame()
        stor_df["id"] = [storage.id]
        stor_df["c_rate"] = [storage.c_rate]
        stor_df["volume"] = [storage.volume]
        stor_df["efficiency"] = [storage.efficiency]
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

    volumes_worth.loc[len(volumes_worth), ["volume", "worth"]] = 0, 0
    if buy_or_sell_side == "buyer":
        df = volumes_worth.sort_values("volume", ascending=True)
    elif buy_or_sell_side == "seller":
        df = volumes_worth.sort_values("volume", ascending=True)
    else:
        raise ValueError("buy_or_sell_side has to be either 'buyer' or 'seller'")

    df = df.diff().dropna().reset_index(drop=True).abs()
    df.rename(columns={"worth": "marginal_price"}, inplace=True)
    return df
