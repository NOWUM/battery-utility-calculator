# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from battery_utility_calculator import Storage
from battery_utility_calculator.uncertainty import (
    DEFAULT_PERSISTENCE_HOURS,
    Scenario,
    build_correlation_matrix,
    calculate_storage_worth_distribution,
    sample_scenarios,
    summarize_worth_distribution,
)

idx_long = pd.date_range("2025-01-01", freq="h", periods=240)
idx_3 = pd.date_range("2025-01-01", freq="h", periods=3)


def flat_inputs(index: pd.DatetimeIndex) -> dict:
    """Constant base series, so every deviation is caused by the noise alone."""
    return dict(
        demand=pd.Series(1.0, index=index),
        solar_generation=pd.Series(1.0, index=index),
        supplier_prices=pd.Series(0.30, index=index),
        eeg_prices=pd.Series(0.08, index=index),
        wholesale_market_prices=pd.Series(0.10, index=index),
        community_market_prices={"aachen": pd.Series(0.15, index=index)},
    )


def deviations(
    scenarios: list[Scenario], attribute: str, location: str = "aachen"
) -> np.ndarray:
    """Flattened deviations from the base level of one quantity."""

    def series(scenario: Scenario) -> pd.Series:
        value = getattr(scenario, attribute)
        # community_market_prices is keyed by location
        return value[location] if isinstance(value, dict) else value

    stacked = np.array(
        [series(scenario).to_numpy(dtype=float) for scenario in scenarios]
    )
    return (stacked - stacked.mean()).ravel()


def test_sample_scenarios_returns_requested_number_and_shape():
    scenarios = sample_scenarios(**flat_inputs(idx_3), n_scenarios=7, seed=1)

    assert len(scenarios) == 7
    for scenario in scenarios:
        assert isinstance(scenario, Scenario)
        assert scenario.demand.index.equals(idx_3)
        assert set(scenario.community_market_prices) == {"aachen"}
        assert scenario.community_market_prices["aachen"].index.equals(idx_3)


def test_sample_scenarios_is_reproducible_and_seed_dependent():
    first = sample_scenarios(**flat_inputs(idx_3), n_scenarios=3, seed=42)
    same = sample_scenarios(**flat_inputs(idx_3), n_scenarios=3, seed=42)
    other = sample_scenarios(**flat_inputs(idx_3), n_scenarios=3, seed=43)

    assert np.allclose(first[0].demand, same[0].demand)
    assert not np.allclose(first[0].demand, other[0].demand)


def test_eeg_prices_are_passed_through_unchanged():
    inputs = flat_inputs(idx_3)
    scenarios = sample_scenarios(**inputs, n_scenarios=5, seed=1)

    for scenario in scenarios:
        assert np.allclose(scenario.eeg_prices, inputs["eeg_prices"])


def test_relative_std_scales_the_spread():
    narrow = sample_scenarios(
        **flat_inputs(idx_long), n_scenarios=60, relative_std={"demand": 0.05}, seed=7
    )
    wide = sample_scenarios(
        **flat_inputs(idx_long), n_scenarios=60, relative_std={"demand": 0.20}, seed=7
    )

    narrow_std = deviations(narrow, "demand").std()
    wide_std = deviations(wide, "demand").std()

    assert wide_std > narrow_std
    assert np.isclose(wide_std / narrow_std, 4.0, rtol=0.1)


def test_zero_relative_std_leaves_a_quantity_untouched():
    inputs = flat_inputs(idx_3)
    scenarios = sample_scenarios(
        **inputs, n_scenarios=4, relative_std={"demand": 0.0}, seed=3
    )

    for scenario in scenarios:
        assert np.allclose(scenario.demand, inputs["demand"])


def test_pv_noise_is_relative_so_zero_generation_stays_zero():
    inputs = flat_inputs(idx_3)
    inputs["solar_generation"] = pd.Series([0.0, 2.0, 0.0], index=idx_3)

    scenarios = sample_scenarios(**inputs, n_scenarios=20, seed=5)

    for scenario in scenarios:
        assert scenario.solar_generation.iloc[0] == 0.0
        assert scenario.solar_generation.iloc[2] == 0.0
        assert (scenario.solar_generation >= 0).all()


def test_price_noise_is_absolute_so_negative_prices_stay_disturbed():
    inputs = flat_inputs(idx_long)
    inputs["wholesale_market_prices"] = pd.Series(-0.05, index=idx_long)

    scenarios = sample_scenarios(**inputs, n_scenarios=40, seed=11)

    spread = deviations(scenarios, "wholesale_market_prices").std()
    assert spread > 0
    # a purely multiplicative model would scale with |−0.05| only; the absolute
    # shift is derived from the mean magnitude and stays meaningful
    assert np.isclose(spread, 0.30 * 0.05, rtol=0.15)


def test_wholesale_and_solar_are_weakly_negatively_correlated():
    # only weakly: wind can depress prices without any sunshine
    scenarios = sample_scenarios(**flat_inputs(idx_long), n_scenarios=120, seed=2)

    correlation = np.corrcoef(
        deviations(scenarios, "wholesale_market_prices"),
        deviations(scenarios, "solar_generation"),
    )[0, 1]

    assert np.isclose(correlation, -0.20, atol=0.06)


def test_supplier_follows_wholesale_exactly():
    scenarios = sample_scenarios(**flat_inputs(idx_long), n_scenarios=120, seed=2)

    correlation = np.corrcoef(
        deviations(scenarios, "wholesale_market_prices"),
        deviations(scenarios, "supplier_prices"),
    )[0, 1]

    assert np.isclose(correlation, 1.0, atol=1e-6)


def test_supplier_inherits_every_wholesale_correlation():
    # with a perfect wholesale/supplier coupling the supplier disturbance IS the
    # wholesale disturbance, so it cannot have relationships of its own
    scenarios = sample_scenarios(**flat_inputs(idx_long), n_scenarios=120, seed=2)

    for other in ("solar_generation", "community_market_prices", "demand"):
        against_wholesale = np.corrcoef(
            deviations(scenarios, "wholesale_market_prices"),
            deviations(scenarios, other),
        )[0, 1]
        against_supplier = np.corrcoef(
            deviations(scenarios, "supplier_prices"),
            deviations(scenarios, other),
        )[0, 1]
        assert np.isclose(against_wholesale, against_supplier, atol=1e-6), other


def test_community_is_nearly_independent_of_wholesale():
    scenarios = sample_scenarios(**flat_inputs(idx_long), n_scenarios=120, seed=2)

    correlation = np.corrcoef(
        deviations(scenarios, "wholesale_market_prices"),
        deviations(scenarios, "community_market_prices"),
    )[0, 1]

    assert np.isclose(correlation, 0.10, atol=0.06)


def test_community_reacts_strongly_to_local_pv():
    scenarios = sample_scenarios(**flat_inputs(idx_long), n_scenarios=120, seed=2)

    correlation = np.corrcoef(
        deviations(scenarios, "community_market_prices"),
        deviations(scenarios, "solar_generation"),
    )[0, 1]

    assert np.isclose(correlation, -0.50, atol=0.06)


def test_default_correlations_are_sampleable():
    # a perfect pair makes the matrix singular; the eigendecomposition has to
    # cope with it where a Cholesky factorisation would fail
    matrix = build_correlation_matrix()
    smallest = float(np.linalg.eigvalsh(matrix).min())

    assert smallest > -1e-8
    assert np.isclose(smallest, 0.0, atol=1e-8)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(matrix)


def test_demand_is_uncorrelated_with_prices_by_default():
    scenarios = sample_scenarios(**flat_inputs(idx_long), n_scenarios=120, seed=2)

    correlation = np.corrcoef(
        deviations(scenarios, "demand"),
        deviations(scenarios, "wholesale_market_prices"),
    )[0, 1]

    assert abs(correlation) < 0.06


def test_correlations_can_be_overridden():
    scenarios = sample_scenarios(
        **flat_inputs(idx_long),
        n_scenarios=120,
        correlations={("demand", "wholesale_market_prices"): 0.6},
        seed=2,
    )

    correlation = np.corrcoef(
        deviations(scenarios, "demand"),
        deviations(scenarios, "wholesale_market_prices"),
    )[0, 1]

    assert np.isclose(correlation, 0.6, atol=0.06)


def test_persistence_shows_up_as_autocorrelation():
    scenarios = sample_scenarios(
        **flat_inputs(idx_long),
        n_scenarios=80,
        persistence_hours=DEFAULT_PERSISTENCE_HOURS,
        hours_per_timestep=1,
        seed=4,
    )

    series = np.array(
        [
            scenario.wholesale_market_prices.to_numpy(dtype=float)
            for scenario in scenarios
        ]
    )
    centered = series - series.mean()
    lag_1 = np.corrcoef(centered[:, :-1].ravel(), centered[:, 1:].ravel())[0, 1]

    expected = np.exp(-1.0 / DEFAULT_PERSISTENCE_HOURS)
    assert np.isclose(lag_1, expected, atol=0.05)


def test_zero_persistence_gives_white_noise():
    scenarios = sample_scenarios(
        **flat_inputs(idx_long), n_scenarios=80, persistence_hours=0.0, seed=4
    )

    series = np.array(
        [
            scenario.wholesale_market_prices.to_numpy(dtype=float)
            for scenario in scenarios
        ]
    )
    centered = series - series.mean()
    lag_1 = np.corrcoef(centered[:, :-1].ravel(), centered[:, 1:].ravel())[0, 1]

    assert abs(lag_1) < 0.05


def test_scenario_to_ecc_kwargs_matches_the_optimizer_signature():
    import inspect

    from battery_utility_calculator import EnergyCostCalculator

    scenario = sample_scenarios(**flat_inputs(idx_3), n_scenarios=1, seed=1)[0]
    parameters = inspect.signature(EnergyCostCalculator.__init__).parameters

    for name in scenario.to_ecc_kwargs():
        assert name in parameters


def test_build_correlation_matrix_is_symmetric_with_unit_diagonal():
    matrix = build_correlation_matrix()

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_build_correlation_matrix_accepts_either_key_order():
    forward = build_correlation_matrix({("demand", "supplier_prices"): 0.3})
    reversed_ = build_correlation_matrix({("supplier_prices", "demand"): 0.3})

    assert np.allclose(forward, reversed_)


def test_build_correlation_matrix_rejects_unknown_quantity():
    with pytest.raises(ValueError, match="Unknown quantity"):
        build_correlation_matrix({("demand", "gas_prices"): 0.3})


def test_build_correlation_matrix_rejects_out_of_range_value():
    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        build_correlation_matrix({("demand", "supplier_prices"): 1.5})


def test_build_correlation_matrix_rejects_impossible_combination():
    # A perfectly follows B and B perfectly follows C, so A and C cannot be
    # anti-correlated at the same time
    with pytest.raises(ValueError, match="not positive semi-definite"):
        build_correlation_matrix(
            {
                ("demand", "supplier_prices"): 1.0,
                ("supplier_prices", "wholesale_market_prices"): 1.0,
                ("demand", "wholesale_market_prices"): -1.0,
            }
        )


def test_sample_scenarios_rejects_mismatched_index():
    inputs = flat_inputs(idx_3)
    inputs["demand"] = pd.Series(
        1.0, index=pd.date_range("2026-01-01", freq="h", periods=3)
    )

    with pytest.raises(ValueError, match="indices must be identical"):
        sample_scenarios(**inputs, n_scenarios=2, seed=1)


def test_sample_scenarios_rejects_non_datetime_index():
    inputs = flat_inputs(idx_3)
    inputs["solar_generation"] = pd.Series([1.0, 1.0, 1.0])

    with pytest.raises(TypeError, match="DateTimeIndex"):
        sample_scenarios(**inputs, n_scenarios=2, seed=1)


def test_sample_scenarios_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="n_scenarios"):
        sample_scenarios(**flat_inputs(idx_3), n_scenarios=0, seed=1)

    with pytest.raises(ValueError, match="must not be negative"):
        sample_scenarios(
            **flat_inputs(idx_3), n_scenarios=2, relative_std={"demand": -0.1}, seed=1
        )

    with pytest.raises(ValueError, match="Unknown quantity"):
        sample_scenarios(
            **flat_inputs(idx_3), n_scenarios=2, relative_std={"gas": 0.1}, seed=1
        )


def test_sample_scenarios_without_community_market():
    inputs = flat_inputs(idx_3)
    inputs["community_market_prices"] = None

    scenarios = sample_scenarios(**inputs, n_scenarios=3, seed=1)

    assert all(scenario.community_market_prices is None for scenario in scenarios)


def worth_inputs(index: pd.DatetimeIndex) -> dict:
    """Base case where a storage has a real value: cheap early, expensive later."""
    return dict(
        demand=pd.Series(1.0, index=index),
        solar_generation=pd.Series(0.0, index=index),
        supplier_prices=pd.Series([0.0, 1.0, 1.0], index=index),
        eeg_prices=pd.Series(0.0, index=index),
        wholesale_market_prices=pd.Series(0.0, index=index),
        community_market_prices=None,
    )


def worth_scenarios(n_scenarios: int = 12, seed: int = 3) -> list[Scenario]:
    return sample_scenarios(
        **worth_inputs(idx_3),
        n_scenarios=n_scenarios,
        relative_std={"supplier_prices": 0.25, "demand": 0.15},
        seed=seed,
    )


def run_distribution(scenarios: list[Scenario], **kwargs) -> pd.DataFrame:
    return calculate_storage_worth_distribution(
        baseline_storage=Storage(0, 1, 0, 1, 1),
        storages_to_calculate=[Storage(1, 1, 1, 1, 1), Storage(2, 1, 2, 1, 1)],
        scenarios=scenarios,
        solver="appsi_highs",
        allow_community_to_home=False,
        allow_community_to_storage=False,
        allow_pv_to_community=False,
        allow_storage_to_community=False,
        **kwargs,
    )


def test_worth_distribution_has_a_row_per_scenario_and_storage():
    scenarios = worth_scenarios(n_scenarios=5)

    distribution = run_distribution(scenarios)

    # baseline plus two storages in each of the five scenarios
    assert len(distribution) == 5 * 3
    assert sorted(distribution["scenario"].unique()) == [0, 1, 2, 3, 4]
    assert set(distribution["volume"]) == {0.0, 1.0, 2.0}
    assert distribution.columns[0] == "scenario"
    for column in ("id", "volume", "cashflow", "worth", "location"):
        assert column in distribution.columns


def test_worth_distribution_keeps_the_baseline_at_zero_worth():
    distribution = run_distribution(worth_scenarios(n_scenarios=6))

    baseline = distribution[distribution["volume"] == 0.0]
    assert len(baseline) == 6
    assert np.allclose(baseline["worth"], 0.0)


def test_worth_distribution_actually_varies_between_scenarios():
    distribution = run_distribution(worth_scenarios(n_scenarios=15))

    worths = distribution.loc[distribution["volume"] == 1.0, "worth"]
    assert worths.std() > 0
    # a bigger storage is never worth less than a smaller one
    for scenario in distribution["scenario"].unique():
        rows = distribution[distribution["scenario"] == scenario].set_index("volume")
        assert rows.loc[2.0, "worth"] >= rows.loc[1.0, "worth"] - 1e-9


def test_worth_distribution_is_deterministic_for_a_fixed_seed():
    first = run_distribution(worth_scenarios(n_scenarios=4, seed=9))
    again = run_distribution(worth_scenarios(n_scenarios=4, seed=9))

    assert np.allclose(first["worth"], again["worth"])


def test_worth_distribution_passes_options_through():
    scenarios = worth_scenarios(n_scenarios=4)

    without_cost = run_distribution(scenarios)
    with_cost = run_distribution(scenarios, cycle_cost_per_kwh=0.2)

    without = without_cost.loc[without_cost["volume"] == 1.0, "worth"].to_numpy()
    with_ = with_cost.loc[with_cost["volume"] == 1.0, "worth"].to_numpy()
    assert (with_ < without).all()


def test_worth_distribution_reports_the_storage_location():
    scenarios = worth_scenarios(n_scenarios=3)

    distribution = run_distribution(
        scenarios,
        my_location="aachen",
        storage_location="liege",
        is_rented_storage=True,
    )

    assert set(distribution["location"]) == {"liege"}


def test_worth_distribution_rejects_empty_and_wrong_types():
    with pytest.raises(ValueError, match="must not be empty"):
        run_distribution([])

    with pytest.raises(TypeError, match="has to be a Scenario"):
        run_distribution(["not a scenario"])


def test_summarize_worth_distribution_aggregates_per_storage():
    distribution = run_distribution(worth_scenarios(n_scenarios=20))

    summary = summarize_worth_distribution(distribution)

    assert len(summary) == 3
    assert (summary["n_scenarios"] == 20).all()
    for column in (
        "worth_mean",
        "worth_std",
        "worth_min",
        "worth_max",
        "cashflow_mean",
    ):
        assert column in summary.columns
    assert list(summary["volume"]) == [0.0, 1.0, 2.0]

    for _, row in summary.iterrows():
        assert row["worth_min"] <= row["worth_q05"] <= row["worth_q50"]
        assert row["worth_q50"] <= row["worth_q95"] <= row["worth_max"]
        assert row["worth_min"] <= row["worth_mean"] <= row["worth_max"]


def test_summarize_worth_distribution_matches_manual_quantiles():
    distribution = run_distribution(worth_scenarios(n_scenarios=20))

    summary = summarize_worth_distribution(distribution, quantiles=[0.1, 0.9])
    one_kwh = distribution.loc[distribution["volume"] == 1.0, "worth"]
    row = summary[summary["volume"] == 1.0].iloc[0]

    assert np.isclose(row["worth_mean"], one_kwh.mean())
    assert np.isclose(row["worth_std"], one_kwh.std())
    assert np.isclose(row["worth_q10"], one_kwh.quantile(0.1))
    assert np.isclose(row["worth_q90"], one_kwh.quantile(0.9))


def test_summarize_worth_distribution_names_fractional_quantiles():
    distribution = run_distribution(worth_scenarios(n_scenarios=6))

    summary = summarize_worth_distribution(distribution, quantiles=[0.025, 0.5])

    assert "worth_q2.5" in summary.columns
    assert "worth_q50" in summary.columns


def test_summarize_worth_distribution_handles_a_single_scenario():
    distribution = run_distribution(worth_scenarios(n_scenarios=1))

    summary = summarize_worth_distribution(distribution)

    assert (summary["n_scenarios"] == 1).all()
    # pandas reports NaN for the std of one sample
    assert np.allclose(summary["worth_std"], 0.0)


def test_summarize_worth_distribution_validates_input():
    distribution = run_distribution(worth_scenarios(n_scenarios=3))

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        summarize_worth_distribution(distribution, quantiles=[1.5])

    with pytest.raises(KeyError, match="missing the columns"):
        summarize_worth_distribution(distribution.drop(columns="worth"))
