#!usr/bin/env python

"""
ABOUT: This module is responsible for downloading from web & loading
into working memory sub-sets of datasets needed.
"""

from source import *

# External libraries
from pathlib import Path
import pandas as pd
import urllib.request
import glob


# Locally downloads specific time slice parts of this dataset. This is huge & dynamic
def download_transit_history(out_dir: str = DATA_DIR, year: str = None, month: str = None):
    for url in sorted(list(pd.read_csv('data/on_time_performance_index.tsv', sep='\t')['URL']), reverse=True):
        # hacky!
        filename = Path(url).name
        file_year = filename.split('_')[-2]
        file_month = filename.split('_')[-1].split('.zip')[0]

        if year in {file_year, None} and month in {file_month, None}:
            print(f"downloading: {filename}")
            urllib.request.urlretrieve(url, f'{out_dir}/{filename}')
    return


# Loads a collection of transit history zipped parts as a memory geo dataframe.
def load_transit_history(
    out_dir: str = DATA_DIR,                # location of downloaded zip files.
    year: str = None, month: str = None,    # parts to load.
    frac: float = 1.0                       # option to sub-sample a fraction per part
) -> pd.DataFrame:
    return pd.concat((
        pd.read_csv(file, compression='zip', header=0, sep=',')
            .sample(frac=frac, random_state=settings.get('random_seed'))
        for file in glob.glob(f"{out_dir}/on_time_performance_{year or '*'}_{month or '*'}.zip")
    ), ignore_index=True)


# Adjusts types, removes irrelevant columns & empty rows.
def clean_transit_history(df: pd.DataFrame) -> pd.DataFrame:
    df.drop([
        'Row ID',       # redundant since reindex
        'Route Name',   # redundant from 'Route Number'
        'Day Type',     # derivable from scheduled date
    ], axis=1, inplace=True)
    df.dropna(inplace=True)

    df['Route Destination'] = df['Route Destination'].astype('category')
    df['Stop Number'] = df['Stop Number'].astype('category')
    df['Deviation'] = df['Deviation'] / 60  # [minutes] neg = late, pos = early
    df['Scheduled Time'] = pd.to_datetime(df['Scheduled Time'], format='ISO8601')
    return df


# Downloads road network from government of canada. This remains zip & includes en/fr
# versions of roads, junctions, entryways, etc. User filtering required.
def download_manitoba_roadnetwork(out_dir: str = DATA_DIR):
    url = 'https://geo.statcan.gc.ca/nrn_rrn/mb/nrn_rrn_mb_SHAPE.zip'
    filename = Path(url).name
    print(f"downloading: {filename}")
    urllib.request.urlretrieve(url, f'{out_dir}/{filename}')
    return


# Downloads latest place from open street map. This remains zip & includes various
# types of point & polygon places. User filtering required.
def download_manitoba_pois(outdir: str = DATA_DIR):
    url = 'http://download.geofabrik.de/north-america/canada/manitoba-latest-free.shp.zip'
    filename = Path(url).name
    print(f"downloading: {filename}")
    urllib.request.urlretrieve(url, f'{outdir}/{filename}')
    return
