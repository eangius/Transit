#!usr/bin/env python

# Internal libraries
from source.data_loaders import *
from source.dataset_builder import *

# Configure pandas to not truncate (...) row & column displays.
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


"""
# Download only last 2 years of data to to keep analysis focused on relevant.
download_transit_history(year="2023")
download_transit_history(year="2022")
"""


# load the most resent data (need all to preserve name sequence structure)
df = clean_transit_history(
    load_transit_history(frac=1.0,  year="2023", month="05")
)

# decouple all static spatial stop information for later easy lookup.
# this is equivalent to our vocabulary & embeddings
# TODO: enrich with more attributes
df_spatial_stops = df[["Location", "Stop Number"]].set_index("Stop Number")
df.drop("Location", axis=1, inplace=True)

# TODO:
# -- should produce a shapefile
# -- should identify directional routes
# -- should identify stop order
# map directional routes to set of stops they contain.
# this is equivalent to our document/term matrix.
# df_spatial_routes = df_raw \
#     .reset_index() \
#     .drop("Scheduled Time", axis=1)[["Trip", "Stop Number"]] \
#     .drop_duplicates() \
#     .value_counts() \
#     .unstack(fill_value=0)
# <<dbg should be a sequence of name rather than sparse
# <<dbg not ideal if routes change path during the day or week


# simplify & combine name & destination into directional routes
df["Route"] = df.apply(lambda row: str(row["Route Number"]) + "__" + str(row["Route Destination"]), axis=1)
df.drop(["Route Number", "Route Destination"], axis=1, inplace=True)

# prepare remaining historical data for machine learning
# 1st split chronologically past & future, then split each into individual
# trips.
df_old, df_new = dataset_splitter(df, column='Scheduled Time', test_frac=0.20, gap_frac=0.05)
df_learn = spatial_route_context(df_old, offset=datetime.timedelta(hours=2))
df_eval = spatial_route_context(df_new, offset=datetime.timedelta(hours=2))
del df, df_old, df_new


# TODO:
# [0] stationirize & de seasonilize
# [1] convert schedule/observations into directional-name-trip matrix
#     -- each cell (stop-id, trip#) cell consists of (scheduled, deviation) feature data.
#     -- padded to longest name in dataset.
# [2] spatial model learns from rows of the matrix
# [3] temporal model learns from columns of the matrix
# [4] each model will featurize the (stop-id, trip#) the same
#     ie: expand scheduled into temporal features
#     ie: expand stop-id into spatial features (lookup)
#     ie: expand trip#

