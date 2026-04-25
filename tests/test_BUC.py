# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

from battery_utility_calculator.battery_utility_calculator import (
    Storage,
    calculate_bidding_curve,
    calculate_multiple_storage_worth,
    calculate_storage_worth,
    plot_multiple_storage_worth_cashflows,
)

idx = pd.date_range("2025-01-01", freq="h", periods=3)


def test_calculate_storage_worth():
    baseline_storage = Storage(0, 1, 0, 1)
    storage_to_calc = Storage(0, 1, 1, 1)

    worth = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices=pd.Series([0, 0, 0], index=idx),
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        solver="appsi_highs",
    )

    # ask for cashflows as well
    result = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices=pd.Series([0, 0, 0], index=idx),
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        return_cashflows=True,
        solver="appsi_highs",
    )
    assert "baseline_cashflows" in result and "storage_to_calc_cashflows" in result
    assert "baseline_soc_ts" not in result and "storage_to_calc_soc_ts" not in result
    # baseline and storage supplier costs should differ by roughly the worth (≈1)
    diff = (
        result["storage_to_calc_cashflows"]["supplier"]
        - result["baseline_cashflows"]["supplier"]
    )
    assert round(diff, 0) == 1

    # with baseline storage costs should be 2
    # with storage costs should be 1
    assert round(worth, 0) == 1

    soc_result = calculate_storage_worth(
        baseline_storage=baseline_storage,
        storage_to_calculate=storage_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices=pd.Series([0, 0, 0], index=idx),
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        return_soc_timeseries=True,
        solver="appsi_highs",
    )
    assert "baseline_soc_ts" in soc_result and "storage_to_calc_soc_ts" in soc_result


def test_calculate_multiple_storage_worth():
    baseline_storage = Storage(0, 1, 0, 1)
    storages_to_calc = [Storage(0, 1, 1, 1), Storage(1, 1, 2, 1)]

    worths = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices=pd.Series([0, 0, 0], index=idx),
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        solver="appsi_highs",
    )

    # request cashflow output too
    df_with_cf = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices=pd.Series([0, 0, 0], index=idx),
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        return_cashflows=True,
        solver="appsi_highs",
    )
    assert "baseline_cashflows" in df_with_cf
    assert isinstance(df_with_cf["storages_to_calc_cashflows"], dict)

    fig = plot_multiple_storage_worth_cashflows(df_with_cf, show=False, stacked=False)
    assert len(fig.data) == 4

    with pytest.raises(ValueError, match="missing"):
        plot_multiple_storage_worth_cashflows({"results_df": df_with_cf["results_df"]})

    df_with_soc = calculate_multiple_storage_worth(
        baseline_storage=baseline_storage,
        storages_to_calculate=storages_to_calc,
        eeg_prices=pd.Series([0, 0, 0], index=idx),
        wholesale_market_prices=pd.Series([0, 0, 0], index=idx),
        community_market_prices=pd.Series([0, 0, 0], index=idx),
        supplier_prices=pd.Series([0, 1, 1], index=idx),
        solar_generation=pd.Series([0, 0, 0], index=idx),
        demand=pd.Series([1, 1, 1], index=idx),
        return_soc_timeseries=True,
        solver="appsi_highs",
    )
    assert "baseline_soc_ts" in df_with_soc
    assert isinstance(df_with_soc["storages_to_calc_soc_ts"], dict)

    print(worths["costs"])
    assert (worths["costs"].round(0).values == [-2, -1, 0]).all()
    assert (worths["worth"].round(0).values[1:] == [1, 2]).all()


def test_calc_bid_curve_dtypes():
    input_df = pd.DataFrame()
    input_df["volume"] = [1, 0]
    input_df["worth"] = [10, 0]
    input_df["other"] = ["A", "B"]

    curve = calculate_bidding_curve(input_df, "buyer")


def test_calculate_bidding_curve_buyer():
    volume_worths = pd.DataFrame()
    volume_worths["volume"] = [0, 1, 2, 3, 3.5]
    volume_worths["worth"] = [0, 5, 7, 8, 8.2]

    bidding_curve = calculate_bidding_curve(
        volumes_worth=volume_worths, buy_or_sell_side="buyer"
    )

    correct_df = pd.DataFrame()
    correct_df["volume"] = [1, 1, 1, 0.5]
    correct_df["marginal_price"] = [5, 2, 1, 0.2]
    correct_df["cumulative_volume"] = [1, 2, 3, 3.5]
    correct_df["marginal_price_per_kwh"] = [5, 2, 1, 0.4]

    assert (bidding_curve["volume"] == correct_df["volume"]).all()
    assert (
        bidding_curve["marginal_price"].round(2)
        == correct_df["marginal_price"].round(2)
    ).all()
    assert (bidding_curve["cumulative_volume"] == correct_df["cumulative_volume"]).all()
    assert (
        bidding_curve["marginal_price_per_kwh"].round(2)
        == correct_df["marginal_price_per_kwh"].round(2)
    ).all()


def test_calculate_bidding_curve_seller():
    volume_worths = pd.DataFrame()
    volume_worths["volume"] = [0, 2, 10]
    volume_worths["worth"] = [-10, -7, 0]

    bidding_curve = calculate_bidding_curve(
        volumes_worth=volume_worths, buy_or_sell_side="seller"
    )

    correct_df = pd.DataFrame()
    correct_df["volume"] = [8, 2]
    correct_df["marginal_price"] = [7, 3]
    correct_df["cumulative_volume"] = [8, 10]
    correct_df["marginal_price_per_kwh"] = [7 / 8, 3 / 2]

    assert (bidding_curve["volume"] == correct_df["volume"]).all()
    assert (
        bidding_curve["marginal_price"].round(2)
        == correct_df["marginal_price"].round(2)
    ).all()
    assert (bidding_curve["cumulative_volume"] == correct_df["cumulative_volume"]).all()
    assert (
        bidding_curve["marginal_price_per_kwh"].round(2)
        == correct_df["marginal_price_per_kwh"].round(2)
    ).all()
