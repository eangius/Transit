#!usr/bin/env python

# External libraries
from typing import *
from datetime import timedelta
import pandas as pd


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


# ABOUT:
# -- takes a dataset of individual bus stop arrival times
# -- groups data into individual routes to get stop sequence order
# -- splits route observations into individual bus trips (based on threshold time gap)
# -- adds deviation times of previous window number of stops (if available)
# -- returns dataset of all observations
def dataset_spatial_context(
    df: pd.DataFrame,  # raw observations ["Scheduled Time", "Stop Number", "Deviation", "Route"]
    window: int = 10,  # number of stops behind current to look back
    pad: int = 0,      # value to set target deviations at start of route.
) -> pd.DataFrame:
    results = []

    offset = timedelta(hours=2)
    for route_name, df_route in df.groupby("Route", observed=False):
        grouper_by_time_gap = df_route.groupby(
            (df_route["Scheduled Time"].diff() > offset).cumsum(),
            observed=True
        )
        results.extend(
            pd.DataFrame([
                {
                    **{
                        k: row[k]
                        for k in {"Scheduled Time", "Route", "Stop Number", "Location"}
                    },

                    # add deviations of previous stops (if any)
                    **{
                        f"Deviation{i * -1}": df_trip["Deviation"].iloc[idx - i] if idx - i >= 0 else pad
                        for i in reversed(range(window + 1))
                    },
                }
                for idx, row in df_trip.reset_index(drop=True).iterrows()
            ])
            for _, df_trip in grouper_by_time_gap
        )
    return pd.concat(results)
