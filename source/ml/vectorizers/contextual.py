#!usr/bin/env python

# Internal libraries
from source.ml.vectorizers import ScikitVectorizer

# External libraries
import warnings
from overrides import overrides
from cachetools import cached, LRUCache
import numpy as np
import time


class RouteInfoVectorizer(ScikitVectorizer):
    """
    Learns structure of distinct routes & the stops sequence they are
    composed of.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._route_stops = dict()  # seq of stops per directional route
        return

    '''
    ASSUMPTIONS:
    [0] each route direction is treated as a separate path.
        while busses can cycle back, we do this to simplify things.

    [1] directional routes do not loop (or have dead end paths) through a common stop
        ok since stops are at different sides of the street.
    
    [2] directional routes always have the same sequence of stops.
        ie: no rush hour express skipping/altering stops.
        
    [3] no staggered frequency of trips within routes. ***
        ie: if multiple busses depart within a few minutes, the temporal
        stop order is broken!
    '''
    # X is a dataframe of shape (n_samples, 3) with ["Scheduled Time", "Route", "Stop Number"] columns
    @overrides
    def fit(self, X, y=None):
        df = X.copy()
        df["Order"] = df["Scheduled Time"].apply(
            lambda dt: time.mktime(dt.timetuple())
        )
        df.drop_duplicates(inplace=True)
        df.sort_values('Order', inplace=True)
        df.set_index('Order', inplace=True)

        # directional route to sequence of stops
        self._route_stops = {
            route: df_grp["Stop Number"].drop_duplicates().to_numpy().ravel()
            for route, df_grp in df.groupby("Route", observed=True)
            if df_grp.shape[0] > 0
        }
        # TODO:
        # -- refactor out column name references
        # -- add distance heuristic?
        return self

    @overrides
    def transform(self, X, y=None):
        return np.array(list(X.apply(
            lambda row: [
                self.route_length(row["Route"]),
                self.route_position(row["Route"], row["Stop Number"])
            ],
            axis=1
        )))

    def get_feature_names_out(self, _unused_input_features=None):
        return np.array([
            self.route_length.__name__,
            self.route_position.__name__,
        ])

    # Absolute size of a route.
    @cached(cache=LRUCache(maxsize=5))
    def route_length(self, route_name: str) -> int:
        return len(self._route_stops.get(route_name, []))

    # percent position of stop within the route (0% = unrecognized, 0% > start, 100% = finish)
    def route_position(self, route_name: str, stop_num: str) -> float:
        stops = self._route_stops.get(route_name, np.array([]))
        indices = np.where(stops == stop_num)[0]
        if indices.size == 0:
            return 0  # unrecognized. may need retraining

        if indices.size > 1:
            warnings.warn(
                f"Only the first position of the {indices.size} found instances " +
                f"for stop {stop_num} of route '{route_name}' will be used."
            )

        size = self.route_length(route_name)
        return ((indices[0] + 1) / size) if size > 0 else 0


'''
#dbg>> code computes distance matrix of stops within a route.
from geopy import distance
import itertools

def meter_distance(pt1, pt2) -> float:
    return distance.distance(
        [pt1.y, pt1.x],
        [pt2.y, pt2.x]
    ).m

df = X_eval.copy()
df["Order"] = df["Scheduled Time"].apply(
    lambda dt: time.mktime(dt.timetuple())
)
df.drop_duplicates(inplace=True)
df.sort_values('Order', inplace=True)

df.set_index('Order', inplace=True)
grouper = iter(df.groupby("Route", observed=True))
route, df_grp = next(grouper)

# compute all distinct stop pair distance combinations
dist_mtrx = {
    (stop1, stop2): meter_distance(pt1, pt2)
    for (stop1, pt1), (stop2, pt2) in itertools.combinations(df_grp[["Stop Number", "Location"]].values, 2)
}
'''
