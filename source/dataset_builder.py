#!usr/bin/env python

# Internal libraries

# External libraries
from typing import *
from dataclasses import dataclass
import pandas as pd


# Chronologically splits time-series into train & test parts. Ensures
# testing on newer & non-overlapping training set with optional gap
# to avoid info leakage.
def temporal_dataset_splitter(
    df: pd.DataFrame,     # data to split
    column: str,          # datetime column
    test_frac: float,     # % of data in test
    gap_frac: float = 0,  # optional % of data to ignore
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if not (0 < test_frac + gap_frac <= 1):
        raise ValueError("Invalid split fractions.")

    df.set_index(column, inplace=True)
    df.sort_index(inplace=True)

    n = df.shape[0]

    split_idx = round(n * (1 - (test_frac + gap_frac)))
    df_train = df.iloc[0:split_idx, :].reset_index()

    split_idx = round(n * (1 - test_frac))
    df_test = df.iloc[split_idx:, :].reset_index()

    return df_train, df_test


@dataclass
class Trip:
    route: str          # route name & direction
    df: pd.DataFrame    # sequence of scheduled & deviation of stops


# NOTE: assumes all trips have the same sequence & number of stops.
# this may not be the case for rush-hour express routes under the same name!
def temporal_trip_context(
    df: pd.DataFrame,
) -> Iterable[Trip]:
    return [
        Trip(route=route, df=df_trips)

        # split observations by daily routes
        for (_, route), df_grp in df.groupby([
            pd.Grouper(key='Scheduled Time', freq='1D', sort=True), "Route"
        ])

        # split daily routes by repeating trips (via 1st stop sequence)
        for _, df_trips in \
            df_grp
                .drop("Route", axis=1)
                .set_index("Scheduled Time")
                .groupby(
                    df_grp['Stop Number'].eq(df_grp['Stop Number'].iloc[0]).cumsum()
                )
    ]
