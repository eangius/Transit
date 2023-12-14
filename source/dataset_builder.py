#!usr/bin/env python

# Internal libraries

# External libraries
from typing import *
from dataclasses import dataclass
import pandas as pd
import datetime


# Chronologically splits time-series into train & test parts. Ensures
# testing on newer & non-overlapping training set with optional gap
# to avoid info leakage.
def dataset_splitter(
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
    name: str         # name name & direction
    df: pd.DataFrame  # sequence of scheduled & deviation of stops


# Groups observations as a collection of daily name trips
# where each is the sequence of stop status
# shape = (name, date, trips, stops, 2)
def spatial_route_context(
    df: pd.DataFrame,  # raw observations ["Scheduled Time", "Stop Number", "Deviation", "Route"]
    offset: datetime.timedelta,
) -> Iterable[Trip]:
    result = []
    for route_name, df_routes in df.groupby("Route"):
        grouper_by_timegap = df_routes.groupby(
            (df_routes["Scheduled Time"].diff() > offset).cumsum()
        )

        df_routes.drop("Route", axis=1, inplace=True)
        df_routes.sort_values("Scheduled Time", inplace=True)
        df_routes.set_index("Scheduled Time", inplace=True)

        # TODO
        # must still split out staggered busses on the same name
        # knowing the routes stop ordering.

        result.extend(
            Trip(route_name, df_trip)
            for _, df_trip in grouper_by_timegap
        )
    return result


# Groups observations as a collection of daily name stops
# where each is the history of trip status for each stop
def temporal_route_context(
    df_raw: pd.DataFrame,  # raw observations ["Scheduled Time", "Stop Number", "Deviation", "Route"]
):
    raise NotImplementedError
