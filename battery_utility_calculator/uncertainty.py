# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

"""Parametric scenario generation for the uncertainty analysis.

The optimizer is deterministic and has perfect foresight. To say something about
how uncertain a storage worth is, the same problem is solved once per sampled
scenario and the resulting spread is evaluated (wait-and-see).

Scenarios are drawn from the deterministic base timeseries by adding correlated
noise. Two properties matter for the result and are therefore explicit:

* Cross correlation. Retail prices follow the exchange, PV weighs on it, and the
  community market reacts mostly to local PV feed-in. Sampling the series
  independently would create combinations that do not occur and would misstate
  the storage worth. See ``DEFAULT_CORRELATIONS`` for the assumed structure.
* Persistence. Forecast errors last - a cloudy day stays cloudy. Independent
  noise per timestep would average out over the horizon and understate the
  spread of the storage worth considerably.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# the order fixes rows and columns of the correlation matrix
UNCERTAIN_QUANTITIES = (
    "demand",
    "solar_generation",
    "supplier_prices",
    "wholesale_market_prices",
    "community_market_prices",
)

# quantities scale with their level, prices are shifted; see the module docstring
_DISTURBANCE_KIND = {
    "demand": "relative",
    "solar_generation": "relative",
    "supplier_prices": "absolute",
    "wholesale_market_prices": "absolute",
    "community_market_prices": "absolute",
}

# rough starting points, meant to be overridden with numbers from actual forecasts
DEFAULT_RELATIVE_STD = {
    "demand": 0.10,
    "solar_generation": 0.20,
    "supplier_prices": 0.10,
    "wholesale_market_prices": 0.30,
    "community_market_prices": 0.30,
}

# Pairwise correlations; unlisted pairs default to 0.0.
#
# Reasoning behind the values:
#
# * Retail prices track the exchange one to one, so their disturbance is the same
#   disturbance. As a consequence supplier_prices is not free to have its own
#   relationship with anything else - it inherits every correlation of
#   wholesale_market_prices. The two supplier entries below are therefore not
#   independent choices, they are copies of the wholesale row, and changing one
#   without the other makes the matrix unsamplable.
# * PV weighs on the exchange price, but only weakly: wind can depress prices just
#   as well on an overcast day, so a low price is a poor indicator of sunshine.
# * The community market is driven by local PV feed-in rather than by the
#   exchange - hence nearly independent of wholesale, but clearly negative
#   against solar_generation.
#
# Note that a single common market factor cannot produce this structure: tying
# supplier to wholesale at 1.0 would force correlation(solar, community) to be
# the product of their wholesale correlations, about -0.02 instead of -0.50. The
# matrix is therefore given directly and checked in build_correlation_matrix.
DEFAULT_CORRELATIONS = {
    ("wholesale_market_prices", "supplier_prices"): 1.00,
    ("wholesale_market_prices", "solar_generation"): -0.20,
    ("wholesale_market_prices", "community_market_prices"): 0.10,
    # forced by the perfect wholesale/supplier coupling above
    ("supplier_prices", "solar_generation"): -0.20,
    ("supplier_prices", "community_market_prices"): 0.10,
    ("community_market_prices", "solar_generation"): -0.50,
}

DEFAULT_PERSISTENCE_HOURS = 12.0


@dataclass(frozen=True)
class Scenario:
    """One complete, internally consistent set of optimizer inputs."""

    demand: pd.Series
    solar_generation: pd.Series
    supplier_prices: pd.Series
    eeg_prices: pd.Series
    wholesale_market_prices: pd.Series
    community_market_prices: dict[str, pd.Series] | None = None

    def to_ecc_kwargs(self) -> dict:
        """Keyword arguments for ``EnergyCostCalculator`` and the worth helpers."""
        return {
            "demand": self.demand,
            "solar_generation": self.solar_generation,
            "supplier_prices": self.supplier_prices,
            "eeg_prices": self.eeg_prices,
            "wholesale_market_prices": self.wholesale_market_prices,
            "community_market_prices": self.community_market_prices,
        }


def build_correlation_matrix(
    correlations: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """Build the correlation matrix over ``UNCERTAIN_QUANTITIES``.

    Args:
        correlations: Pairwise correlations keyed by a tuple of quantity names in
            either order. Unlisted pairs are uncorrelated. ``None`` uses
            ``DEFAULT_CORRELATIONS``.

    Returns:
        Symmetric matrix with ones on the diagonal.

    Raises:
        ValueError: On unknown quantity names, correlations outside [-1, 1],
            contradicting entries for the same pair, or a matrix that is not
            positive semi-definite and therefore not sampleable.
    """
    if correlations is None:
        correlations = DEFAULT_CORRELATIONS

    index = {name: position for position, name in enumerate(UNCERTAIN_QUANTITIES)}
    matrix = np.eye(len(UNCERTAIN_QUANTITIES))

    for (first, second), value in correlations.items():
        for name in (first, second):
            if name not in index:
                msg = (
                    f"Unknown quantity '{name}' in correlations. "
                    f"Allowed values: {sorted(index)}"
                )
                raise ValueError(msg)
        if not -1.0 <= value <= 1.0:
            msg = f"Correlation for ('{first}', '{second}') must be within [-1, 1], got {value}."
            raise ValueError(msg)
        if first == second:
            if value != 1.0:
                msg = f"Correlation of '{first}' with itself must be 1.0, got {value}."
                raise ValueError(msg)
            continue

        row, column = index[first], index[second]
        existing = matrix[row, column]
        if existing != 0.0 and existing != value:
            msg = (
                f"Contradicting correlations given for '{first}' and '{second}': "
                f"{existing} and {value}."
            )
            raise ValueError(msg)
        matrix[row, column] = value
        matrix[column, row] = value

    smallest_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
    if smallest_eigenvalue < -1e-8:
        msg = (
            "The correlation matrix is not positive semi-definite (smallest "
            f"eigenvalue {smallest_eigenvalue:.4f}), so no joint distribution "
            "matches it. Relax the conflicting pairs, e.g. a strong positive "
            "correlation of A with B and of B with C forces A and C to be "
            "correlated as well."
        )
        raise ValueError(msg)

    return matrix


def _persistence_factor(persistence_hours: float, hours_per_timestep: float) -> float:
    """AR(1) coefficient for an e-folding time of ``persistence_hours``."""
    if persistence_hours <= 0:
        return 0.0
    return float(np.exp(-hours_per_timestep / persistence_hours))


def _correlated_ar1_noise(
    n_scenarios: int,
    n_timesteps: int,
    correlation_matrix: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Standard normal noise, correlated across quantities and over time.

    Returns an array of shape ``(n_scenarios, n_timesteps, len(UNCERTAIN_QUANTITIES))``
    with unit marginal variance, so the caller only has to scale it.
    """
    n_quantities = correlation_matrix.shape[0]

    # eigendecomposition rather than Cholesky, so that merely semi-definite
    # matrices (a correlation of exactly 1.0 between two quantities) still work
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))

    white = rng.standard_normal((n_scenarios, n_timesteps, n_quantities))
    correlated = white @ factor.T

    # AR(1) along the time axis; the scaling keeps the marginal variance at one,
    # which also preserves the cross correlation imposed above
    noise = np.empty_like(correlated)
    noise[:, 0, :] = correlated[:, 0, :]
    innovation_scale = float(np.sqrt(1.0 - rho**2))
    for timestep in range(1, n_timesteps):
        noise[:, timestep, :] = (
            rho * noise[:, timestep - 1, :]
            + innovation_scale * correlated[:, timestep, :]
        )

    return noise


def _disturb(base: pd.Series, noise: np.ndarray, relative_std: float, kind: str):
    """Apply one quantity's noise column to its base timeseries."""
    values = base.to_numpy(dtype=float)

    if kind == "relative":
        # scales with the level, so a night without PV stays a night without PV
        disturbed = values * (1.0 + relative_std * noise)
        disturbed = np.clip(disturbed, 0.0, None)
    else:
        # absolute shift derived from the mean magnitude, which keeps working
        # for prices at or below zero
        scale = relative_std * float(np.abs(values).mean())
        disturbed = values + scale * noise

    return pd.Series(disturbed, index=base.index, name=base.name)


def _check_indices(series_by_name: dict[str, pd.Series]) -> pd.Index:
    reference_name, reference = next(iter(series_by_name.items()))
    if not isinstance(reference.index, pd.DatetimeIndex):
        msg = f"Index of {reference_name} has to be pd.DateTimeIndex!"
        raise TypeError(msg)

    for name, series in series_by_name.items():
        if not isinstance(series.index, pd.DatetimeIndex):
            msg = f"Index of {name} has to be pd.DateTimeIndex!"
            raise TypeError(msg)
        if not series.index.equals(reference.index):
            msg = (
                "All timeseries indices must be identical. Index of "
                f"{name} does not equal index of {reference_name}."
            )
            raise ValueError(msg)

    return reference.index


def sample_scenarios(
    demand: pd.Series,
    solar_generation: pd.Series,
    supplier_prices: pd.Series,
    eeg_prices: pd.Series,
    wholesale_market_prices: pd.Series,
    community_market_prices: dict[str, pd.Series] | None = None,
    *,
    n_scenarios: int = 100,
    relative_std: dict[str, float] | None = None,
    correlations: dict[tuple[str, str], float] | None = None,
    persistence_hours: float = DEFAULT_PERSISTENCE_HOURS,
    hours_per_timestep: int | float = 1,
    seed: int | None = None,
) -> list[Scenario]:
    """Draw scenarios around deterministic base timeseries.

    ``demand`` and ``solar_generation`` are disturbed multiplicatively and clipped
    at zero, prices are shifted by an absolute amount derived from their mean
    magnitude so that negative prices keep working. Disturbances are correlated
    across quantities and follow an AR(1) process over time.

    ``eeg_prices`` are passed through unchanged - the feed-in tariff is fixed by
    law and carries no forecast uncertainty.

    All locations in ``community_market_prices`` share one disturbance, since
    neighbouring local markets are assumed to move together. Model separate
    location risk by sampling those series yourself and assembling
    :class:`Scenario` objects directly.

    Args:
        demand: Base demand in kW per timestep, with a ``pd.DatetimeIndex``.
        solar_generation: Base PV generation in kW per timestep.
        supplier_prices: Base grid prices in EUR per kWh.
        eeg_prices: EEG prices in EUR per kWh, used unchanged.
        wholesale_market_prices: Base wholesale prices in EUR per kWh.
        community_market_prices: Base community prices per location, or ``None``.
        n_scenarios: Number of scenarios to draw.
        relative_std: Relative standard deviation per quantity, overriding
            ``DEFAULT_RELATIVE_STD``. For prices it refers to the mean magnitude
            of the respective series.
        correlations: Pairwise correlations, see :func:`build_correlation_matrix`.
        persistence_hours: E-folding time of the disturbances in hours. Larger
            values make forecast errors last longer; ``0`` gives white noise.
        hours_per_timestep: Duration of one timestep in hours, used to translate
            ``persistence_hours`` into the AR(1) coefficient.
        seed: Seed for reproducible draws.

    Returns:
        A list of ``n_scenarios`` :class:`Scenario` objects.

    Raises:
        TypeError: If an index is not a ``pd.DatetimeIndex``.
        ValueError: On mismatched indices, ``n_scenarios < 1``, negative standard
            deviations, unknown quantity names or a non-sampleable correlation matrix.
    """
    if n_scenarios < 1:
        msg = f"n_scenarios has to be at least 1, got {n_scenarios}."
        raise ValueError(msg)

    base_series = {
        "demand": demand,
        "solar_generation": solar_generation,
        "supplier_prices": supplier_prices,
        "eeg_prices": eeg_prices,
        "wholesale_market_prices": wholesale_market_prices,
    }
    if community_market_prices is not None:
        for location, series in community_market_prices.items():
            base_series[f"community_market_prices['{location}']"] = series
    index = _check_indices(base_series)

    standard_deviations = dict(DEFAULT_RELATIVE_STD)
    if relative_std is not None:
        for name, value in relative_std.items():
            if name not in DEFAULT_RELATIVE_STD:
                msg = (
                    f"Unknown quantity '{name}' in relative_std. "
                    f"Allowed values: {sorted(DEFAULT_RELATIVE_STD)}"
                )
                raise ValueError(msg)
            if value < 0:
                msg = f"relative_std['{name}'] must not be negative, got {value}."
                raise ValueError(msg)
            standard_deviations[name] = float(value)

    correlation_matrix = build_correlation_matrix(correlations)
    rho = _persistence_factor(float(persistence_hours), float(hours_per_timestep))
    rng = np.random.default_rng(seed)
    noise = _correlated_ar1_noise(
        n_scenarios=n_scenarios,
        n_timesteps=len(index),
        correlation_matrix=correlation_matrix,
        rho=rho,
        rng=rng,
    )

    column = {name: position for position, name in enumerate(UNCERTAIN_QUANTITIES)}
    scenarios = []
    for scenario_number in range(n_scenarios):
        scenario_noise = noise[scenario_number]

        def disturb(name: str, base: pd.Series, scenario_noise=scenario_noise):
            return _disturb(
                base=base,
                noise=scenario_noise[:, column[name]],
                relative_std=standard_deviations[name],
                kind=_DISTURBANCE_KIND[name],
            )

        if community_market_prices is None:
            sampled_community = None
        else:
            sampled_community = {
                location: disturb("community_market_prices", series)
                for location, series in community_market_prices.items()
            }

        scenarios.append(
            Scenario(
                demand=disturb("demand", demand),
                solar_generation=disturb("solar_generation", solar_generation),
                supplier_prices=disturb("supplier_prices", supplier_prices),
                eeg_prices=eeg_prices.copy(),
                wholesale_market_prices=disturb(
                    "wholesale_market_prices", wholesale_market_prices
                ),
                community_market_prices=sampled_community,
            )
        )

    return scenarios
