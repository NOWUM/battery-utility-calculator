# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT


class Storage:
    def __init__(self, id: int, c_rate: float, volume: float, efficiency: float):
        """Represents a storage unit for energy.

        Args:
            id (int): The unique identifier for the storage unit.
            c_rate (float): The charging rate of the storage unit (kWh/h).
            volume (float): The total capacity of the storage unit (kWh).
            efficiency (float): The efficiency of the storage unit (0-1).
        """
        self.id = id
        self.c_rate = c_rate
        self.volume = volume
        self.efficiency = efficiency
