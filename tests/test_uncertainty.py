# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from battery_utility_calculator.uncertainty import (
    DEFAULT_PERSISTENCE_HOURS,
    Scenario,
    build_correlation_matrix,
    sample_scenarios,
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
