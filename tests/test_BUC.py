# SPDX-FileCopyrightText: NOWUM Developers
#
# SPDX-License-Identifier: MIT

import pandas as pd

from battery_utility_calculator.battery_utility_calculator import (
    calculate_bidding_curve,
)


def test_calculate_bidding_curve():
    volume_worths = pd.DataFrame()
    volume_worths["volume"] = [1, 2, 3]
    volume_worths["worth"] = [5, 7, 8]

    bidding_curve = calculate_bidding_curve(
        volumes_worth=volume_worths, buy_or_sell_side="buyer"
    )

    correct_df = pd.DataFrame()
    correct_df["volume"] = [1, 1, 1]
    correct_df["marginal_price"] = [5, 2, 1]

    assert (bidding_curve == correct_df).all().all()
