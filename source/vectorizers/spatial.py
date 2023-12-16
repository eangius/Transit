#!usr/bin/env python


# Internal libraries
from source.vectorizers.frequential import ItemCountVectorizer

# External libraries
from shapely.geometry.point import Point
from typing import Set
import h3.api.numpy_int as h3
import numpy as np


class GeoVectorizer(ItemCountVectorizer):
    """
    Converts shapely latitude & longitude point coordinates to a geospatial indexing
    scheme. This is useful to quantize areas & model neighboring or hierarchical spatial
    relationships between them. Some relationships are 1:1 but others are 1:many, so
    resulting vectors denote occurrence counts of all train-time known areas. Any
    unrecognized area at inference time is ignored & vectorized as zero. Depending on
    spatial resolution & coverage, vectors can be high-dimensional & are encoded as
    sparse. Users may want to cap dimensionality by size, frequency or perform other
    dimensionality reduction techniques.
    """

    def __init__(
        self,
        resolution: int,           # cell size of this area (range depends on scheme)
        index_scheme: str = 'h3',  # geo indexing scheme
        items: Set[str] = None,    # combo of 'cell', 'neighbor', 'parent' or 'children'.
        offset: int = 1,           # neighbouring or hierarchical cells away from this
        **kwargs                   # see ItemCountVectorizer inputs.
    ):
        if index_scheme != 'h3':
            # TODO: implement geohash, s2, ..
            raise NotImplemented(
                f"Unrecognized indexing schem {self.indexing_scheme}"
            )

        self.resolution = resolution
        self.items = items or {'cells'}
        self.offset = offset
        self.index_scheme = index_scheme
        super().__init__(**kwargs)
        return

    def fit(self, X, y=None):
        return super().fit(self._convert(X), y)

    def transform(self, X, y=None):
        return super().transform(self._convert(X), y)

    # approximate coordinates from area centroids
    def inverse_transform(self, X):
        return np.array([
            h3.h3_to_geo(y) if y else self.out_of_vocab
            for y in super().inverse_transform(X)
        ])

    def _convert(self, X):
        # accumulates item types into the same vector
        items = []
        if 'cells' in self.items:
            items.extend(list(map(self._cells, X)))
        if 'neighbors' in self.items:
            items.extend(list(map(self._neighbors, X)))
        if 'parents' in self.items:
            items.extend(list(map(self._parents, X)))
        if 'children' in self.items:
            items.extend(list(map(self._children, X)))
        return np.array(items).reshape(-1, 1)

    def _cells(self, geom: Point):
        return h3.geo_to_h3(geom.y, geom.x, self.resolution)

    def _neighbors(self, geom: Point):
        return h3.hex_ring(self._cells(geom), self.setps)

    def _parents(self, geom: Point):
        return h3.h3_to_parent(self._cells(geom), self.resolution - self.offset)

    def _children(self, geom: Point):
        return h3.h3_to_children(self._cells(geom), self.resolution + self.offset)
