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

## Example

- Create a Storage object describing capacity and charge/discharge limits.
- Provide price, PV generation and demand time series (as pandas DataFrame/Series).
- Construct a `PricingFramework`, call `optimize(...)` with a solver (e.g. "highs") and inspect results via the model or `output_results()`.


The following example shows that a storage with a volume of 1 kWh reduces the cost to buy the given demand by 1 as the storage is used to charge in cheap times.

```python
import pandas as pd
from battery_utility_calculator.pricing_framework import PricingFramework, Storage

# now we need 2 kWh at each timestep
# on timestep=0, we can buy for 0€/kWh and should buy 3kWh
# as we use 2 kWh during timestep=0 and use 1 kWh for timestep=1
# total cost should be 3*0 + 1*1 + 2*1 = 3
pricer = PricingFramework(
    storage=Storage(id=0, c_rate=1, volume=1, efficiency=1),
    prices=pd.DataFrame(
        {
            "eeg": [0, 0, 0],
            "wholesale": [0, 0, 0],
            "community": [0, 0, 0],
            "grid": [0, 1, 1],
        }
    ),
    solar_generation=pd.Series([0, 0, 0]),
    demand=pd.Series([2, 2, 2]),
)
pricer.optimize(solver="highs")
assert pricer.model.objective() == -3

# or get timeseries output (after optimization)
results = pricer.output_results()
```

The cost of the optimal dispatch timeseries is provided in `pricer.model.objective()`

## License
MIT - see [LICENSE](./LICENSE)

