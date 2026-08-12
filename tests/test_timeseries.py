# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from battery_utility_calculator import EnergyCostCalculator, Storage, sample_scenarios
from battery_utility_calculator.timeseries import (
    check_identical_indices,
    describe_index_mismatch,
    indices_match,
)

idx_ns = pd.date_range("2025-01-01", freq="h", periods=6, tz="UTC")


def test_indices_match_ignores_the_datetime_resolution():
    # Index.equals compares the dtype too, so this pair comes out unequal on
    # pandas 2.x although both describe the very same instants. pandas 3 already
    # tolerates it, which means this test only guards the regression on 2.x -
    # the supported range starts there, so the check has to stay.
    idx_us = idx_ns.astype("datetime64[us, UTC]")

    assert indices_match(idx_ns, idx_us)
    assert indices_match(idx_ns, idx_ns.astype("datetime64[s, UTC]"))


def test_indices_match_accepts_an_identical_index():
    assert indices_match(idx_ns, idx_ns.copy())


def test_indices_match_rejects_real_differences():
    assert not indices_match(idx_ns, idx_ns[:5])
    assert not indices_match(idx_ns, idx_ns[::-1])
    assert not indices_match(idx_ns, idx_ns + pd.Timedelta(hours=1))


def test_indices_match_rejects_timezone_differences():
    # a naive index cannot be placed on a timeline, and the same wall clock in
    # another zone denotes different moments
    assert not indices_match(idx_ns, idx_ns.tz_localize(None))
    assert not indices_match(
        idx_ns, idx_ns.tz_localize(None).tz_localize("Europe/Berlin")
    )


def test_indices_match_accepts_the_same_instants_in_another_zone():
    # tz_convert keeps the moments and only changes how they are written down
    converted = idx_ns.tz_convert("Europe/Berlin")

    assert converted[0] != idx_ns[0].tz_localize(None)
    assert indices_match(idx_ns, converted)


def test_indices_match_treats_missing_entries_like_index_equals():
    # NaT never equals itself, so the value comparison needs to handle it
    with_nat = pd.DatetimeIndex(list(idx_ns[:5]) + [pd.NaT], tz="UTC")

    assert indices_match(with_nat, with_nat.astype("datetime64[us, UTC]"))
    assert not indices_match(with_nat, idx_ns)


def test_describe_index_mismatch_names_the_cause():
    shorter = describe_index_mismatch(idx_ns, idx_ns[:5], "demand", "supplier_prices")
    assert "len=6" in shorter and "len=5" in shorter
    assert "only in demand" in shorter

    reordered = describe_index_mismatch(idx_ns, idx_ns[::-1], "demand", "other")
    assert "different order" in reordered

    shifted = describe_index_mismatch(
        idx_ns, idx_ns + pd.Timedelta(hours=1), "demand", "other"
    )
    assert "first difference at position 0" in shifted

    naive = describe_index_mismatch(idx_ns, idx_ns.tz_localize(None), "demand", "other")
    assert "timezones differ" in naive

    # the timezone stays the reported cause even when the length differs too
    both = describe_index_mismatch(
        idx_ns, idx_ns.tz_localize(None)[:4], "demand", "other"
    )
    assert "timezones differ" in both


def test_check_identical_indices_returns_the_reference():
    series = {
        "demand": pd.Series(1.0, index=idx_ns),
        "supplier_prices": pd.Series(0.3, index=idx_ns.astype("datetime64[us, UTC]")),
    }

    assert check_identical_indices(series).equals(idx_ns)


def test_check_identical_indices_reports_a_mismatch():
    series = {
        "demand": pd.Series(1.0, index=idx_ns),
        "supplier_prices": pd.Series(0.3, index=idx_ns[:5]),
    }

    with pytest.raises(ValueError, match="supplier_prices"):
        check_identical_indices(series)


def test_check_identical_indices_requires_a_datetime_index():
    series = {"demand": pd.Series([1.0, 2.0]), "other": pd.Series([1.0, 2.0])}

    with pytest.raises(TypeError, match="DateTimeIndex"):
        check_identical_indices(series)


def mixed_resolution_inputs() -> dict:
    """The failing notebook case: some series in ns, others in us."""
    idx_us = idx_ns.astype("datetime64[us, UTC]")
    return dict(
        demand=pd.Series(1.0, index=idx_ns),
        solar_generation=pd.Series(0.5, index=idx_us),
        supplier_prices=pd.Series(0.32, index=idx_us),
        eeg_prices=pd.Series(0.08, index=idx_ns),
        wholesale_market_prices=pd.Series(0.09, index=idx_us),
        community_market_prices={"aachen": pd.Series(0.18, index=idx_us)},
    )


def test_sample_scenarios_accepts_mixed_resolutions():
    scenarios = sample_scenarios(**mixed_resolution_inputs(), n_scenarios=3, seed=1)

    assert len(scenarios) == 3
    assert len(scenarios[0].demand) == len(idx_ns)


def test_energy_cost_calculator_accepts_mixed_resolutions():
    calculator = EnergyCostCalculator(
        storage=Storage(id=0, c_rate=1, volume=2), **mixed_resolution_inputs()
    )

    objective = calculator.optimize(solver="appsi_highs")

    assert np.isfinite(objective)
    assert len(calculator.get_energy_flows()) == len(idx_ns)
