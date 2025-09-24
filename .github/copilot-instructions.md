<!-- Short, focused guidance for AI coding agents working on this repo -->

# Battery Utility Calculator — Copilot instructions

Goal: Help an AI agent become productive quickly by explaining the project's structure, important data shapes, workflows, and a few known quirks.

1) Big picture
- Single small Python package that models a prosumer (PV + optional battery) energy dispatch optimizer using Pyomo.
- Core logic lives in `battery_utility_calculator/battery_utility_calculator.py` (classes: `Storage`, `BatteryUtilityCalculator`).
- Tests live in `tests/test_pricing_framework.py` and exercise high-level scenarios (they instantiate the calculator with simple pandas objects and call `optimize(...)`).

2) Key files to read first
- `battery_utility_calculator/battery_utility_calculator.py` — main model, variables, constraints, objective, result exporters.
- `tests/test_pricing_framework.py` — concise examples of expected behaviours (use these to create new tests or reproduce issues).
- `pyproject.toml` — dependencies and dev/test matrix (note: `pyomo`, `highspy` and `pandas` are direct deps).
- `README.md` — quick run and install instructions.

3) Data shapes & APIs (concrete, copyable)
- Prices: pandas.DataFrame expected columns: `eeg`, `wholesale`, `community`, `grid`. Code references prices via `self.prices.loc[timestep, "<col>"]`.
- Solar & demand: pandas.Series, aligned with `prices` by index in the intended design.
- Storage: created via `Storage(id: int, c_rate: float, volume: float, efficiency: float)` and passed to `BatteryUtilityCalculator`.
- Example from tests:

```py
prices = pd.DataFrame({"eeg":[0,0,0], "wholesale":[0,0,0], "community":[0,0,0], "grid":[1,1,1]})
solar = pd.Series([0,0,0])
demand = pd.Series([1,1,1])
calc = BatteryUtilityCalculator(Storage(0,1,0,1), prices, solar, demand)
calc.optimize(solver="highs")
```

4) Important implementation patterns and naming conventions
- Model variable names are systematic and map to energy flows: `pv_to_home`, `pv_to_eeg`, `pv_to_storage`, `storage_to_home`, `storage_to_eeg`, `wholesale_to_storage`, `supplier_to_home`, etc.
- `storage_use_cases` is a list of strings (defaults: `["eeg","wholesale","community","home"]`) and is used as a second index for some variables and storage level bookkeeping.
- Objective maximizes aggregated cashflows: community_cf + supplier_cf + eeg_cf + wholesale_cf (see `set_model_objective`).
- Solver is selected by `optimize(solver: str)`; default in code is `gurobi`, but tests use `highs`.

5) Developer workflows (how to run, test, lint)
- Install for development and tests: `pip install -e .[test]` (documented in README).
- Run tests: `pytest` (project has a `tool.pytest.testpaths = "tests"`).
- Linting: ruff configured in `pyproject.toml` — run `ruff .` as needed.

6) Integration & external dependencies to be aware of
- Pyomo models: the code builds a `pyo.ConcreteModel()` and relies on external solvers (e.g., `highs`, commercial solvers like `gurobi`). Ensure the solver you call is available in the environment.
- `highspy` is a listed dependency and provides an open-source solver interface used in tests.

7) Known quirks and places to be careful (important for automated edits)
- Timeseries index expectations: the code defines `__check_timeseries_indices__` (expects pandas DatetimeIndex in UTC), but this method is not called currently — tests use plain integer indices. Do not assume the check is enforced unless you add the call.
- Shape mismatches / TODOs: several result-exporting helpers assume different variable shapes than how variables are declared (e.g., `pv_to_storage` is declared with indexes `(timestep, use_case)` but exporters iterate `(t,use,sid)` or assume `storage.id` is iterable). These are real issues to watch for when adding features or refactoring.
- `Storage.id` is currently a simple int in tests and class; some code treats it like an iterable. Prefer adding a migration/compatibility layer (or change `Storage.id` to always be a list) rather than editing exporters in isolation.
- Default solver string in `optimize()` is `gurobi` — CI/tests call `highs`. Avoid changing defaults without updating tests.

8) Suggested small, safe entry tasks for an AI agent
- Add unit tests reproducing a failing exporter path (e.g., when `storage.id` is a single int vs iterable).
- Add a single call to `__check_timeseries_indices__` early in `__init__` (and update tests immediately if they rely on integer indices) — this is a targeted refactor with clear scope.
- Add validation and conversion helpers that accept integer-index time series and convert to a DatetimeIndex with UTC (if desired) — keep it opt-in.

9) Contacts & conventions
- Tests are the canonical spec for behaviour — follow `tests/test_pricing_framework.py` when in doubt.
- Follow ruff rules defined in `pyproject.toml` for formatting and linting.

If any section is unclear or you want me to expand examples (for example: adding a small test to lock down `storage.id` behavior), tell me which area and I will iterate.
