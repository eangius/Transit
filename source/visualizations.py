#!usr/bin/env python

"""
ABOUT: specific plots & visuals used in the analysis to hide
details from notebook.
"""

# Internal libraries

# External libraries
import matplotlib.pyplot as plt
import pandas as pd
from typing import *
from shapely import wkt, wkb


def plot_timeseries(df: pd.DataFrame):
    df_viz = df[['Scheduled Time', 'Deviation']].copy()
    df_viz['Scheduled Time'] = df_viz['Scheduled Time'].dt.date
    df_viz = df_viz.groupby('Scheduled Time').agg(["mean", "std"])
    df_viz.columns = df_viz.columns.droplevel()
    plt.plot(df_viz['mean'])
    plt.fill_between(
        df_viz['std'].index,
        df_viz['mean'] - df_viz['std'],
        df_viz['mean'] + df_viz['std'],
        color='b', alpha=.1
    )
    plt.title('Daily Average & Standard Bus Arrival Deviation')
    plt.xticks(rotation=90)
    plt.ylabel('minutes')
    plt.tight_layout()


# We plot histograms in QGIS. This triages the data at each location by some
# temporal slicing function & export the grouped data to TSV file.
def export_histogram(df: pd.DataFrame, fn: Callable = None):
    delim = '\t'
    df_viz = df[['Location', 'Scheduled Time', 'Deviation']].copy()
    df_viz['Location'] = df_viz['Location'].apply(lambda wkb_hex: wkb.loads(wkb_hex, hex=True))
    if fn is not None:
        df_viz['Scheduled Time'] = df_viz['Scheduled Time'].apply(fn)
        for cls in set(df_viz['Scheduled Time'].values):
            df_viz[df_viz['Scheduled Time'] == cls] \
                .drop('Scheduled Time', axis=1) \
                .groupby('Location').mean() \
                .to_csv(f'data/locations_{cls}.tsv', sep=delim)
    else:
        df_viz \
            .drop('Scheduled Time', axis=1) \
            .groupby('Location').mean() \
            .to_csv('data/locations_all.tsv', sep=delim, index=True)
    return
