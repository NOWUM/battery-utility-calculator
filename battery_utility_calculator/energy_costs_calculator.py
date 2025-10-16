# SPDX-FileCopyrightText: Christoph Komanns, Florian Maurer, Ralf Schemm
#
# SPDX-License-Identifier: MIT

import logging

import pandas as pd
import plotly.express as px
import pyomo.environ as pyo

from battery_utility_calculator import Storage

log = logging.getLogger("battery_utility")
log.setLevel(logging.WARNING)


class EnergyCostCalculator:
    def __init__(
        self,
        storage: Storage | None,
        demand: pd.Series,
        solar_generation: pd.Series,
        grid_prices: pd.Series,
        eeg_prices: pd.Series,
        community_market_prices: pd.Series,
        wholesale_market_prices: pd.Series,
        storage_use_cases: list[str] = ["eeg", "wholesale", "community", "home"],
        allow_pv_to_community: bool = False,
        allow_storage_to_wholesale: bool = False,
        allow_community_to_home: bool = False,
        allow_community_to_storage: bool = False,
        allow_storage_to_community: bool = False,
    ):
        """Optimizer for prosumer energy management, giving price recommendations for given products and given timeframes

        Args:
            storage (Storage) | None: The available storage. None if no storage is available.
            demand (pd.Series): The demand data for the optimization. Values should be in kWh per hour (kW).
            solar_generation (pd.Series): The solar generation data for the optimization. Values should be in kWh per hour (kW).
            grid_prices (pd.Series): The grid prices for the optimization. Values should be in EUR per kWh.
            eeg_prices (pd.Series): The EEG prices for the optimization. Values should be in EUR per kWh.
            community_market_prices (pd.Series): The community market prices for the optimization. Values should be in EUR per kWh.
            wholesale_market_prices (pd.Series): The wholesale market prices for the optimization. Values should be in EUR per kWh.
            storage_use_cases (list[str]): The use cases for energy storage. Allowed values are "eeg", "wholesale", "community", "home"
        """

        if storage:
            self.storage = storage
        else:
            self.storage = Storage(id=0, c_rate=1, volume=0, efficiency=1)

        self.grid_prices = grid_prices
        self.eeg_prices = eeg_prices
        self.community_market_prices = community_market_prices
        self.wholesale_market_prices = wholesale_market_prices
        self.solar_generation = solar_generation
        self.demand = demand
        self.storage_use_cases = storage_use_cases

        self.allow_pv_to_community = allow_pv_to_community
        self.allow_storage_to_wholesale = allow_storage_to_wholesale
        self.allow_community_to_home = allow_community_to_home
        self.allow_storage_to_community = allow_storage_to_community
        self.allow_community_to_storage = allow_community_to_storage

        self.timesteps = list(range(len(self.demand.index)))

        self.__check_timeseries_indices__()

        self.start = self.demand.index[0]
        self.end = self.demand.index[-1]

        if isinstance(self.demand.index, pd.DatetimeIndex):
            self.timestamps = pd.date_range(
                start=self.start, end=self.end, periods=len(self.demand)
            )
        else:
            self.timestamps = self.timesteps

        self.solar_generation = solar_generation
        self.demand = demand

        self.is_optimized = False

        self.model = pyo.ConcreteModel()
        self.set_model_variables()
        self.set_model_constraints()
        self.set_model_objective()

    def __check_timeseries_indices__(self) -> None:
        """Check if all timeseries indices are valid."""
        attrs = [
            "solar_generation",
            "grid_prices",
            "eeg_prices",
            "community_market_prices",
            "wholesale_market_prices",
        ]

        # reference
        ref_index = self.demand.index
        self.timestamps = ref_index.copy()  # save original index for later
        if isinstance(ref_index, pd.DatetimeIndex):
            ref_index = self.timesteps  # set to integer values
            log.warning("DatetimeIndex for demand not allowed, set to integer")

        # ensure all indices match the reference and if DatetimeIndex
        # are provided convert the actual instance attributes to integer indices
        # while keeping the original timestamps in self.timestamps.
        for name in attrs:
            series = getattr(self, name)
            # compare against saved timestamps (which may be integer list)
            if not series.index.equals(self.timestamps):
                raise ValueError(f"{name}: all timeseries indices must be identical.")

            if isinstance(series.index, pd.DatetimeIndex):
                # convert the actual stored Series index to integer timesteps
                series.index = self.timesteps
                setattr(self, name, series)
                log.warning(f"DatetimeIndex for {name} not allowed, set to integer")

    def set_model_variables(self):
        log.info("Setting up model variables...")

        # Selling PV for EEG
        self.model.pv_to_eeg = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)

        # Selling PV on wholesale
        self.model.pv_to_wholesale = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals
        )

        # Selling PV on community market
        if self.allow_pv_to_community:
            self.model.pv_to_community = pyo.Var(
                self.timesteps,
                domain=pyo.NonNegativeReals,
            )
        else:
            self.model.pv_to_community = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # PV energy to storage
        self.model.pv_to_storage = pyo.Var(
            self.timesteps,
            self.storage_use_cases,
            domain=pyo.NonNegativeReals,
        )

        # using PV energy at home
        self.model.pv_to_home = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)

        # storage level restrction
        self.model.storage_level = pyo.Var(
            self.timesteps,
            self.storage_use_cases,
            domain=pyo.NonNegativeReals,
        )

        # selling from storage for EEG
        self.model.storage_to_eeg = pyo.Var(self.timesteps, domain=pyo.NonNegativeReals)

        # selling from storage on wholesale
        if self.allow_storage_to_wholesale:
            self.model.storage_to_wholesale = pyo.Var(
                self.timesteps,
                domain=pyo.NonNegativeReals,
            )
        else:
            self.model.storage_to_wholesale = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # selling from storage on community
        if self.allow_storage_to_community:
            self.model.storage_to_community = pyo.Var(
                self.timesteps,
                domain=pyo.NonNegativeReals,
            )
        else:
            self.model.storage_to_community = pyo.Var(
                self.timesteps,
                domain=pyo.NonNegativeReals,
                bounds=(0, 0),
            )

        # using energy from storage at home
        self.model.storage_to_home = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals
        )

        # charging storage from wholesale market
        self.model.wholesale_to_storage = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals
        )

        # charging storage from community market
        # remove bounds for activation
        if self.allow_community_to_storage:
            self.model.community_to_storage = pyo.Var(
                self.timesteps,
                domain=pyo.NonNegativeReals,
            )
        else:
            self.model.community_to_storage = pyo.Var(
                self.timesteps,
                domain=pyo.NonNegativeReals,
                bounds=(0, 0),
            )

        # buying energy from community market
        if self.allow_community_to_home:
            self.model.community_to_home = pyo.Var(
                self.timesteps,
                domain=pyo.NonNegativeReals,
            )
        else:
            self.model.community_to_home = pyo.Var(
                self.timesteps, domain=pyo.NonNegativeReals, bounds=(0, 0)
            )

        # charging storage from supplier
        self.model.supplier_to_storage = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals
        )

        # buying energy from supplier
        self.model.supplier_to_home = pyo.Var(
            self.timesteps, domain=pyo.NonNegativeReals
        )

        log.info("Model variables set up successfully.")

    def set_model_constraints(self):
        log.info("Setting up model constraints...")

        # consumption must equal supply (from PV system, supplier, community market, PV system)
        def restrict_demand(model, timestep):
            return (
                model.storage_to_home[timestep]
                + model.pv_to_home[timestep]
                + model.supplier_to_home[timestep]
                + model.community_to_home[timestep]
                == self.demand.iloc[timestep]
            )

        self.model.demand_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_demand
        )

        # use of PV system must be smaller or equal than PV system generation
        def restrict_solar_gen(model, timestep):
            return (
                sum(
                    model.pv_to_storage[timestep, use] for use in self.storage_use_cases
                )
                + model.pv_to_eeg[timestep]
                + model.pv_to_home[timestep]
                + model.pv_to_wholesale[timestep]
                + model.pv_to_community[timestep]
                <= self.solar_generation.iloc[timestep]
            )

        self.model.solar_gen_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_solar_gen
        )

        # energy flow TO storage must be smaller than c_rate
        def restrict_storage_charge(model, timestep):
            return (
                sum(
                    model.pv_to_storage[timestep, use] for use in self.storage_use_cases
                )
                + model.wholesale_to_storage[timestep]
                + model.community_to_storage[timestep]
                + model.supplier_to_storage[timestep]
                <= self.storage.c_rate * self.storage.volume
            )

        self.model.storage_charge_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_storage_charge
        )

        # energy flow FROM storage must be smaller than c_rate
        def restrict_storage_discharge(model, timestep):
            return (
                +model.storage_to_eeg[timestep]
                + model.storage_to_wholesale[timestep]
                + model.storage_to_community[timestep]
                + model.storage_to_home[timestep]
                <= self.storage.c_rate * self.storage.volume
            )

        self.model.storage_discharge_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_storage_discharge
        )

        # storage level must be smaller than volume
        def restrict_soc_max(model, timestep):
            return (
                sum(
                    model.storage_level[timestep, use] for use in self.storage_use_cases
                )
                <= self.storage.volume
            )

        self.model.storage_level_restriction = pyo.Constraint(
            self.timesteps, rule=restrict_soc_max
        )

        # storage level must be larger than 0
        def restrict_soc_min(model, timestep):
            return (
                sum(
                    model.storage_level[timestep, use] for use in self.storage_use_cases
                )
                >= 0
            )

        self.model.storage_level_non_negative = pyo.Constraint(
            self.timesteps, rule=restrict_soc_min
        )

        if "eeg" in self.storage_use_cases:

            def restrict_soc_eeg(model, timestep):
                if timestep == self.timesteps[0]:
                    return (
                        model.storage_level[timestep, "eeg"]
                        == model.pv_to_storage[timestep, "eeg"]
                        - model.storage_to_eeg[timestep]
                    )
                else:
                    previous_timestep = timestep - 1
                    return (
                        model.storage_level[timestep, "eeg"]
                        == model.storage_level[previous_timestep, "eeg"]
                        + model.pv_to_storage[timestep, "eeg"]
                        - model.storage_to_eeg[timestep]
                    )

            self.model.soc_eeg_restriction = pyo.Constraint(
                self.timesteps, rule=restrict_soc_eeg
            )

        if "wholesale" in self.storage_use_cases:

            def restrict_soc_wholesale(model, timestep):
                if timestep == self.timesteps[0]:
                    return (
                        model.storage_level[timestep, "wholesale"]
                        == model.wholesale_to_storage[timestep]
                        - model.storage_to_wholesale[timestep]
                    )
                else:
                    previous_timestep = timestep - 1
                    return (
                        model.storage_level[timestep, "wholesale"]
                        == model.storage_level[previous_timestep, "wholesale"]
                        + model.wholesale_to_storage[timestep]
                        - model.storage_to_wholesale[timestep]
                    )

            self.model.soc_wholesale_restriction = pyo.Constraint(
                self.timesteps, rule=restrict_soc_wholesale
            )

        if "community" in self.storage_use_cases:

            def restrict_soc_community(model, timestep):
                if timestep == self.timesteps[0]:
                    return (
                        model.storage_level[timestep, "community"]
                        == model.community_to_storage[timestep]
                        - model.storage_to_community[timestep]
                    )
                else:
                    previous_timestep = timestep - 1
                    return (
                        model.storage_level[timestep, "community"]
                        == model.storage_level[previous_timestep, "community"]
                        + model.community_to_storage[timestep]
                        - model.storage_to_community[timestep]
                    )

            self.model.soc_community_restriction = pyo.Constraint(
                self.timesteps, rule=restrict_soc_community
            )

        if "home" in self.storage_use_cases:

            def restrict_soc_home(model, timestep):
                if timestep == self.timesteps[0]:
                    return (
                        model.storage_level[timestep, "home"]
                        == model.supplier_to_storage[timestep]
                        + model.pv_to_storage[timestep, "home"]
                        - model.storage_to_home[timestep]
                    )
                else:
                    previous_timestep = timestep - 1
                    return (
                        model.storage_level[timestep, "home"]
                        == model.storage_level[previous_timestep, "home"]
                        + model.supplier_to_storage[timestep]
                        + model.pv_to_storage[timestep, "home"]
                        - model.storage_to_home[timestep]
                    )

            self.model.soc_home_restriction = pyo.Constraint(
                self.timesteps, rule=restrict_soc_home
            )

        log.info("Model constraints set up successfully.")

    def set_model_objective(self):
        log.info("Setting up model objective...")

        # community market cashflow
        community_cf = (
            # selling from storage to community market
            sum(
                self.model.storage_to_community[timestep]
                * self.community_market_prices.loc[timestep]
                for timestep in self.timesteps
            )
            # selling from pv to community market
            + sum(
                self.model.pv_to_community[timestep]
                * self.community_market_prices.loc[timestep]
                for timestep in self.timesteps
            )
            # buying from community market to storage
            - sum(
                self.model.community_to_storage[timestep]
                * self.community_market_prices.loc[timestep]
                for timestep in self.timesteps
            )
            # buying from community market to home
            - sum(
                self.model.community_to_home[timestep]
                * self.community_market_prices.loc[timestep]
                for timestep in self.timesteps
            )
        )

        # supplier cashflow
        supplier_cf = (
            # buying energy from supplier to storage
            -sum(
                self.model.supplier_to_storage[timestep]
                * self.grid_prices.loc[timestep]
                for timestep in self.timesteps
            )
            # buying energy from supplier to home
            - sum(
                self.model.supplier_to_home[timestep] * self.grid_prices.loc[timestep]
                for timestep in self.timesteps
            )
        )

        # EEG cashflow
        eeg_cf = (
            # selling from storage for EEG
            sum(
                self.model.storage_to_eeg[timestep] * self.eeg_prices.loc[timestep]
                for timestep in self.timesteps
            )
            # selling from PV for EEG
            + sum(
                self.model.pv_to_eeg[timestep] * self.eeg_prices.loc[timestep]
                for timestep in self.timesteps
            )
        )

        # wholesale cashflow
        wholesale_cf = (
            # selling from storage to wholesale
            sum(
                self.model.storage_to_wholesale[timestep]
                * self.wholesale_market_prices.loc[timestep]
                for timestep in self.timesteps
            )
            # buying from wholesale to storage
            - sum(
                self.model.wholesale_to_storage[timestep]
                * self.wholesale_market_prices.loc[timestep]
                for timestep in self.timesteps
            )
        )

        # maximize sum of cashflows
        self.model.objective = pyo.Objective(
            expr=community_cf + supplier_cf + eeg_cf + wholesale_cf, sense=pyo.maximize
        )

        log.info("Model objective set up successfully.")

    def optimize(self, solver: str = "gurobi"):
        optimizer = pyo.SolverFactory(
            solver,
        )

        optimizer.solve(self.model, tee=False)

        self.is_optimized = True

        return self.model.objective()

    def get_demand_coverage_timeseries_df(self):
        demand_coverage = pd.DataFrame(index=self.timesteps)

        demand_coverage["demand"] = self.demand.copy()
        demand_coverage["from_grid"] = [
            self.model.supplier_to_home[timestep].value for timestep in self.timesteps
        ]
        demand_coverage["from_pv"] = [
            self.model.pv_to_home[timestep].value for timestep in self.timesteps
        ]
        demand_coverage["from_storage"] = [
            self.model.storage_to_home[timestep].value for timestep in self.timesteps
        ]
        demand_coverage["from_community"] = [
            self.model.community_to_home[timestep].value for timestep in self.timesteps
        ]

        return demand_coverage

    def get_solar_generation_timeseries_df(self):
        pv_usage = pd.DataFrame(index=self.timesteps)

        pv_usage["generation"] = self.solar_generation.loc[self.timesteps]

        pv_usage["to_home"] = [self.model.pv_to_home[t].value for t in self.timesteps]
        pv_usage["to_eeg"] = [self.model.pv_to_eeg[t].value for t in self.timesteps]
        pv_usage["to_community"] = [
            self.model.pv_to_community[t].value for t in self.timesteps
        ]
        pv_usage["to_wholesale"] = [
            self.model.pv_to_wholesale[t].value for t in self.timesteps
        ]
        pv_usage["to_storage_home"] = [
            self.model.pv_to_storage[t, "home"].value for t in self.timesteps
        ]
        pv_usage["to_storage_eeg"] = [
            self.model.pv_to_storage[t, "eeg"].value for t in self.timesteps
        ]
        pv_usage["to_storage_wholesale"] = [
            self.model.pv_to_storage[t, "wholesale"].value for t in self.timesteps
        ]
        pv_usage["to_storage_community"] = [
            self.model.pv_to_storage[t, "community"].value for t in self.timesteps
        ]

        return pv_usage

    def get_storage_timeseries_df(self):
        storage_usage = pd.DataFrame(index=self.timesteps)

        for use_case in self.storage_use_cases:
            storage_usage[f"soc_{use_case}"] = [
                self.model.storage_level[t, use_case].value for t in self.timesteps
            ]

        return storage_usage

    def output_results(self):
        if not self.is_optimized:
            raise ValueError(
                "Model not optimized yet - run class method optimize() first!"
            )

        demand_timeseries = self.get_demand_coverage_timeseries_df()
        pv_timeseries = self.get_solar_generation_timeseries_df()
        storage_timeseries = self.get_storage_timeseries_df()

        return {
            "demand": demand_timeseries,
            "pv": pv_timeseries,
            "storage": storage_timeseries,
        }

    def plot_demand_coverage(self, show: bool = True):
        """Quick stacked bar + demand line using plotly.express."""
        demand_df = self.get_demand_coverage_timeseries_df()

        df = demand_df.reset_index().rename(
            columns={
                demand_df.index.name or "index": "t",
                "from_pv": "From PV",
                "from_storage": "From storage",
                "from_community": "From community",
                "from_grid": "From grid",
            }
        )

        # only plot those with usage
        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])

        # long format for stacking
        supply_cols = [
            c
            for c in ["From PV", "From storage", "From community", "From grid"]
            if c in df.columns
        ]
        long = df.melt(
            id_vars=[df.columns[0], "demand"],
            value_vars=supply_cols,
            var_name="source",
            value_name="kWh",
        )
        fig = px.area(
            long, x=df.columns[0], y="kWh", color="source", title="Demand coverage"
        )

        if "demand" in df.columns:
            fig.add_scatter(x=df[df.columns[0]], y=df["demand"], name="Demand")

        if show:
            fig.show()
        return fig

    def plot_solar_generation(self, show: bool = True):
        """Quick PV generation and usage plot using plotly.express."""
        pv_df = self.get_solar_generation_timeseries_df()

        df = pv_df.reset_index()

        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])

        df = df.rename(
            columns={
                "channel": "Channel",
                "to_home": "To home",
                "to_eeg": "To EEG",
                "to_community": "To community",
                "to_wholesale": "To wholesale",
                "to_storage_home": "To storage for home",
                "to_storage_eeg": "To storage for EEG",
                "to_storage_wholesale": "To storage for wholesale",
                "to_storage_community": "To storage for community",
            }
        )

        usage_cols = [c for c in df.columns if c.startswith("To ")]
        long = df.melt(
            id_vars=[df.columns[0]],
            value_vars=usage_cols,
            var_name="Channel",
            value_name="kWh",
        )
        fig = px.area(
            long,
            x=df.columns[0],
            y="kWh",
            color="Channel",
            title="PV generation & usage",
        )

        if "generation" in df.columns:
            fig.add_scatter(x=df[df.columns[0]], y=df["generation"], name="Generation")

        if show:
            fig.show()
        return fig

    def plot_storage_timeseries(self, show: bool = True):
        """Plot storage SOC per use case using plotly.express."""
        storage_df = self.get_storage_timeseries_df()

        df = storage_df.reset_index().rename(
            columns={
                "soc_home": "Home",
                "soc_eeg": "EEG",
                "soc_wholesale": "Wholesale",
                "soc_community": "Community",
                "use_case": "Use case",
            }
        )

        df = df.drop(columns=[col for col in df.columns if (df[col] == 0).all()])
        long = df.melt(id_vars=[df.columns[0]], var_name="Use case", value_name="kWh")
        fig = px.line(
            long, x=df.columns[0], y="kWh", color="Use case", title="Storage SOC"
        )

        if show:
            fig.show()
        return fig
