<!--
SPDX-FileCopyrightText: NOWUM Developers

SPDX-License-Identifier: MIT
-->

<!-- Short, focused guidance for AI coding agents working on this repo -->

# Battery Utility Calculator — Copilot instructions (concise)

Goal: get an AI agent productive quickly. This repo is a small Python package that models a prosumer (PV + optional battery) dispatch optimizer using Pyomo. Focus on the optimizer, its data inputs, and the tests which are the ground truth.

Core modules

- `battery_utility_calculator/energy_costs_calculator.py` — `EnergyCostCalculator` (builds the Pyomo model, variables, constraints, objective, exporters, plots).
- `battery_utility_calculator/battery_utility_calculator.py` — helper functions (`calculate_storage_worth`, `calculate_multiple_storage_worth`, `calculate_multiple_storage_worth_by_location`, `calculate_bidding_curve`, `plot_multiple_storage_worth_cashflows`) that call ECC.
- `battery_utility_calculator/storage.py` — simple `Storage(id, c_rate, volume, charge_efficiency=0.98, discharge_efficiency=0.98)` value object.

Key tests to read

- `tests/test_BUC.py` and `tests/test_ECC.py` — concise, deterministic examples used as the canonical spec. Copy their small pandas examples when you add new tests or reproduce bugs.

Concrete data shapes & minimal examples

- Every timeseries is a separate `pandas.Series` argument (`demand`, `solar_generation`, `supplier_prices`, `eeg_prices`, `wholesale_market_prices`), not one DataFrame.
- `community_market_prices` is a `dict[str, pd.Series]` keyed by location, or `None` to disable the community market.
- **All series must share one identical `pd.DatetimeIndex`.** A non-datetime index raises `TypeError`, a mismatched one `ValueError`. Do not copy integer-index examples from older docs.
- `demand` and `solar_generation` are kW and get multiplied by `hours_per_timestep` internally; prices are EUR/kWh.
- Example (from tests):

```py
import pandas as pd
from battery_utility_calculator import Storage, calculate_storage_worth

index = pd.date_range("2025-01-01", freq="h", periods=3)
worth = calculate_storage_worth(
    baseline_storage=Storage(id=0, c_rate=1, volume=0, charge_efficiency=1),
    storage_to_calculate=Storage(id=1, c_rate=1, volume=1, charge_efficiency=1),
    eeg_prices=pd.Series([0, 0, 0], index=index),
    wholesale_market_prices=pd.Series([0, 0, 0], index=index),
    community_market_prices={"aachen": pd.Series([0, 0, 0], index=index)},
    supplier_prices=pd.Series([0, 1, 1], index=index),
    solar_generation=pd.Series([0, 0, 0], index=index),
    demand=pd.Series([1, 1, 1], index=index),
    solver="appsi_highs",
    # without these the community market supplies everything at price 0
    # and the storage is worth nothing
    allow_community_to_home=False,
    allow_community_to_storage=False,
    allow_pv_to_community=False,
    allow_storage_to_community=False,
)
```

Important code patterns & conventions

- Variables map to energy flows with systematic names: `pv_to_home`, `pv_to_eeg`, `pv_to_storage`, `storage_to_home`, `storage_to_eeg`, `wholesale_to_storage`, `supplier_to_home`, etc.
- `storage_use_cases` (default `['eeg','wholesale','community','home']`) is used as a second index for `pv_to_storage` and `storage_level` and drives per-use-case SOC constraints.
- Community flow variables carry a second index for the location, e.g. `pv_to_community[t, "aachen"]`.
- Disabled flows are not omitted — they are declared with `bounds=(0, 0)` so the exporters keep working. Follow that pattern when adding a new `allow_*` switch.
- Community storage imports are split: `community_to_storage_for_home` (home SOC, gated by `allow_community_to_storage`) and `community_to_storage_for_community` (community SOC, gated by `allow_community_market_arbitrage`). `allow_storage_to_community` enables discharge from the community SOC (PV via `pv_to_storage['community']` or arbitrage).
- Objective: `__set_max_cashflow_objective__()` maximizes summed cashflows (community + supplier + EEG + wholesale + grid fees) minus the discharge penalty and cycle cost. `set_max_green_energy_objective()` is the alternative for `goal="max_green_energy"`.
- Grid fees come from `grid_fee_between_locations[from][to]`; user values are overlaid on `DEFAULT_GRID_FEE_BETWEEN_LOCATIONS`, so only its four locations are accepted. The table is duplicated in both modules and must stay in sync.
- Solver strings: `ECC.optimize` defaults to `gurobi`; tests call open solvers like `appsi_highs` (or `highs`). Use the exact tester solver string when running tests.

Integration & dependencies

- Pyomo is used for modeling. Solvers must be available in the runtime environment (`highs`, `appsi_highs`, `gurobi`, etc.). `highspy` is a declared dependency.

Project-specific quirks (important for edits)

- Time index handling: `EnergyCostCalculator.__init__` calls `__check_prepare_timeseries_indices__()`, which requires a `DatetimeIndex` on every series, converts them to integer timesteps and keeps the original in `self.original_index`. The exporters use that attribute to restore timestamps.
- Exporters and shape assumptions: exporters assume `storage_use_cases` strings exist and access model variables like `pv_to_storage[t,'home']`. If you change variable indices or use-case names, update all exporters and tests.
- Plot methods take `show: bool = True`; always pass `show=False` in tests and only assert on the returned `go.Figure`.
- `calculate_multiple_storage_worth` raises if two storages share an `id` while any `return_*` flag is set, because ids key the returned dicts.

Developer workflows (quick commands)

- Install dev/test deps and local editable package: `pip install -e .[test]`.
- Run full test suite: `pytest -q` (tests live in `tests/`).
- Run a single test quickly: `pytest -q tests/test_BUC.py::test_calculate_storage_worth`.
- Lint and format before committing: `ruff check .` and `ruff format .` (both run via pre-commit).

Where to look first when debugging

- `battery_utility_calculator/energy_costs_calculator.py`: inspect `__set_model_variables__()`, `set_model_constraints()`, `__set_max_cashflow_objective__()` and `optimize()`.
- Use `tests/` as the authoritative behaviour; copy their minimal pandas structures when reproducing problems.
