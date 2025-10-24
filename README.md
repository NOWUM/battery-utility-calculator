<!--
SPDX-FileCopyrightText: Christoph Komanns, Florian Maurer

SPDX-License-Identifier: MIT
-->

# Battery Utility Calculator

This tool provides a calculation of the utility a storage provides to an electricity consumer.

The utility is calculated by optimizing the storage dispatch and comparing the utility of the storage vs without it.
Iterating over different storage volumes can create a price curve of the stepwise utility each additional capacity provides.
Such values can be used in bidding projects or to investigate whether an additional storage is beneficial.

## Install & run tests

```sh
pip install -e .[test]
pytest
```

## Usage examples (public helpers)

This package exposes small helper functions for common workflows plus the underlying optimizer class.

- `Storage(id, c_rate, volume, efficiency)` — small value object describing a storage unit.
- `calculate_storage_worth(baseline_storage, storage_to_calculate, demand, solar_generation, grid_prices, eeg_prices, community_market_prices, wholesale_market_prices, ...)` — returns the value (difference in optimized costs) of adding `storage_to_calculate` compared to `baseline_storage`.
- `calculate_multiple_storage_worth(...)` — same as above but returns a DataFrame with costs and worth for multiple storage sizes.
- `calculate_bidding_curve(volumes_worth, buy_or_sell_side)` — converts cumulative worths into a marginal bidding curve.

Minimal examples (copied from `tests/`):

```py
import pandas as pd
from battery_utility_calculator.battery_utility_calculator import (
    Storage,
    calculate_storage_worth,
    calculate_multiple_storage_worth,
    calculate_bidding_curve,
)

# single worth
baseline = Storage(0, 1, 0, 1)
candidate = Storage(0, 1, 1, 1)
worth = calculate_storage_worth(
    baseline_storage=baseline,
    storage_to_calculate=candidate,
    eeg_prices=pd.Series([0, 0, 0]),
    wholesale_market_prices=pd.Series([0, 0, 0]),
    community_market_prices=pd.Series([0, 0, 0]),
    grid_prices=pd.Series([0, 1, 1]),
    solar_generation=pd.Series([0, 0, 0]),
    demand=pd.Series([1, 1, 1]),
    solver="appsi_highs",
)

# multiple worths
storages = [Storage(0, 1, 1, 1), Storage(0, 1, 2, 1)]
df = calculate_multiple_storage_worth(
    baseline_storage=baseline,
    storages_to_calculate=storages,
    eeg_prices=pd.Series([0, 0, 0]),
    wholesale_market_prices=pd.Series([0, 0, 0]),
    community_market_prices=pd.Series([0, 0, 0]),
    grid_prices=pd.Series([0, 1, 1]),
    solar_generation=pd.Series([0, 0, 0]),
    demand=pd.Series([1, 1, 1]),
    solver="appsi_highs",
)

# bidding curve
vol_worth = pd.DataFrame({"volume": [1, 2, 3], "worth": [5, 7, 8]})
curve = calculate_bidding_curve(volumes_worth=vol_worth, buy_or_sell_side="buyer")
```

Notes about the optimizer

The core optimizer is `EnergyCostCalculator` (in `battery_utility_calculator/energy_costs_calculator.py`). It builds a Pyomo `ConcreteModel` with variables like `pv_to_storage[t,use]` and per-use-case storage state-of-charge variables. The objective maximizes summed cashflows (community + supplier + EEG + wholesale). If you need lower-level control or plotting, instantiate `EnergyCostCalculator` directly and call `optimize(solver=...)`.

Be aware of time-index handling: the optimizer normalizes timeseries indices to integer timesteps while preserving an original `timestamps` copy when `DatetimeIndex` inputs are used; tests commonly use simple integer-index Series.

## License
MIT - see [LICENSE](./LICENSE)
