#!usr/bin/env python

# Internal libraries

# External libraries
from typing import *
import pandas as pd


# Chronologically splits time-series into train, validate & test parts. Ensures
# test set is newer & non-overlapping the other parts to avoid leakage.
def temporal_dataset_splitter(
    df: pd.DataFrame,   # data to split
    column: str,        # name of the datetime column to chrono
    period: int = 5,    # days in each period
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df.reset_index(inplace=True)
    df.set_index(column, inplace=True)
    groups = df.sort_index().groupby(pd.Grouper(freq=f'{period}D'))

    # TODO:
    # -- implement gap between test
    # -- implement custom ratio
    n_datasets = 3
    d_train, df_validate, df_test = tuple(
        pd.concat((
            df_group if not df_group.empty else pd.DataFrame()
            for i, (*_, df_group) in enumerate(groups) if i % n_datasets == dataset
        ), ignore_index=True)
        for dataset in range(n_datasets)
    )
    return d_train, df_validate, df_test
