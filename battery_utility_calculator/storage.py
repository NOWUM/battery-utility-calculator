# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT


class Storage:
    def __init__(
        self,
        id: int,
        c_rate: float,
        volume: float,
        charge_efficiency: float = 0.98,
        discharge_efficiency: float = 0.98,
    ):
        """Represents a storage unit for energy.

        Args:
            id (int): The unique identifier for the storage unit.
            c_rate (float): C-rate of the storage unit in 1/h, i.e. the fraction of
                ``volume`` that can be charged or discharged per hour. The energy
                limit per timestep is ``c_rate * volume * hours_per_timestep`` in kWh,
                so ``c_rate=1`` moves one full ``volume`` per hour and ``c_rate=0.5``
                half of it. Applies to charging and discharging alike.
            volume (float): The total capacity of the storage unit (kWh).
            charge_efficiency (float): Share of charged energy that arrives in the
                storage (0-1). Charging by ``E`` kWh raises the SOC by
                ``charge_efficiency * E``.
            discharge_efficiency (float): Share of discharged energy that leaves the
                storage (0-1). Delivering ``E`` kWh lowers the SOC by
                ``E / discharge_efficiency``.
        """
        self.id = id
        self.c_rate = c_rate
        self.volume = volume
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
