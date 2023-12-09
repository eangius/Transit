#!usr/bin/env python

"""
ABOUT: specific plots & visuals used in the analysis to hide
details from notebook.
"""

# External libraries
import matplotlib.pyplot as plt
import pandas as pd
import calendar
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
    return


def plot_histogram(df: pd.DataFrame, freq_lbl: str):
    lbls = \
        list(range(24)) if freq_lbl == 'hourly' else \
        list(calendar.day_abbr) if freq_lbl == 'daily' else \
        list(calendar.month_abbr) if freq_lbl == 'monthly' else \
        []

    df_viz = df[['Scheduled Time', 'Deviation']].copy()
    df_viz['Scheduled Time'] = \
        df_viz['Scheduled Time'].dt.hour if freq_lbl == 'hourly' else \
        df_viz['Scheduled Time'].dt.dayofweek if freq_lbl == 'daily' else \
        df_viz['Scheduled Time'].dt.month if freq_lbl == 'monthly' else \
        df_viz['Scheduled Time']
    df_viz['Deviation'] = df_viz['Deviation'].apply(lambda t:
        'delayed' if t < -1 else
        'on-time' if -1 <= t <= 1 else      # assuming +/- 1 min offset is acceptable
        'early'
    )
    df_viz = df_viz.groupby(['Scheduled Time', 'Deviation']).size().to_frame('counts')
    df_viz = df_viz.unstack(level=-1, fill_value=0)
    df_viz.rename(
        index=dict(enumerate(lbls)),
        inplace=True,
        errors='ignore',
    )
    df_viz.columns = df_viz.columns.droplevel()
    df_viz = 100 * df_viz / df_viz.sum(axis='columns').sum()  # relative
    df_viz = df_viz[['delayed', 'on-time', 'early']]          # logical stacking order
    df_viz.plot(
        kind='bar', stacked=True, width=0.9, legend=False,
        title=f'{freq_lbl.capitalize()} Bus Status',
        xlabel='',
        ylabel='% volume of city service',
        rot=0,
        color={
            "delayed": "coral",
            "on-time": "yellowgreen",
            "early": "gold"
        }
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 1, 0]  # swap order to reflect stacking
    plt.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="upper left"
    )
    plt.tight_layout()
    return


# We plot histograms in QGIS. This triages the data at each location by some
# temporal slicing function & export the grouped data to TSV file.
def export_hot_zone(df: pd.DataFrame, fn_temporal: Callable = None):
    delim = '\t'
    df_viz = df[['Location', 'Scheduled Time', 'Deviation']].copy()
    df_viz['Location'] = df_viz['Location'].apply(lambda wkb_hex: wkb.loads(wkb_hex, hex=True))
    if fn_temporal is not None:
        df_viz['Scheduled Time'] = df_viz['Scheduled Time'].apply(fn_temporal)
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
