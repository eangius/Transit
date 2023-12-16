#!usr/bin/env python


# Internal libraries
from source.vectorizers.frequential import ItemCountVectorizer

# External libraries
import h3.api.numpy_int as h3
import numpy as np


class GeoVecorizer(ItemCountVectorizer):
    """
    Converts latitude & longitude point coordinates to a geospatial indexing scheme.
    This is useful to quantize areas & model neighboring or hierarchical spatial
    relationships between them. Some relationships are 1:1 but others are 1:many, so
    resulting vectors denote occurrence counts of all train-time known areas. Any
    unrecognized area at inference time is ignored & vectorized as zero. Depending on
    spatial resolution & coverage, vectors can be high-dimensional & are encoded as
    sparse. Users may want to cap dimensionality by size, frequency or perform other
    dimensionality reduction techniques.
    """

    def __init__(
        self,
        resolution: int,        # hex cell size 0-15
        steps: int = 1,         # neighbouring or up the hierarchy
        geo_index: str = 'h3',  # geo indexing scheme
        mode: str = None,       # mapping to 'cell', 'neighbor', 'parent', 'children'
        **kwargs                # see ItemCountVectorizer inputs.
    ):
        if self.indexing_scheme != 'h3':
            # TODO: implement geohash, s2, ..
            raise NotImplemented(f"Geo indexing schem {self.indexing_scheme} is not supported")

        self.resolution = resolution
        self.steps = steps
        self.mode = mode
        self.geo_index = geo_index
        super().__init__(**kwargs)
        return

    def fit(self, X, y=None):
        return super().fit(self._convert(X), y)

    def transform(self, X, y=None):
        return super().transform(self._convert(X), y)

    def _convert(self, X):
        return np.array(
            list(map(self._cells, X)) if self.mode == 'cells' else
            list(map(self._neighbors, X)) if self.mode == 'neighbors' else
            list(map(self._parents, X)) if self.mode == 'parents' else
            list(map(self._children, X)) if self.mode == 'children' else
            X  # identity
        )

    def _cells(self, lat, lng):
        return h3.geo_to_h3(lat, lng, self.resolution)

    def _neighbors(self, lat, lng):
        return h3.hex_ring(self._cells(lat, lng), self.setps)

    def _parents(self, lat, lng):
        return h3.h3_to_parent(self._cells(lat, lng), self.resolution - self.steps)

    def _children(self, lat, lng):
        return h3.h3_to_children(self._cells(lat, lng), self.resolution - self.steps)
