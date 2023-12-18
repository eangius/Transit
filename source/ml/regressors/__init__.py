#!usr/bin/env python

# External libraries
import numpy as np
from abc import ABC
from scikeras.wrappers import KerasRegressor as RealKerasRegressor
from tensorflow.random import set_seed


# Abstract base class that custom deep-learned models should inherit
# from to integrate into scikit-pipelines.
class KerasRegressor(ABC, RealKerasRegressor):
    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 16,
        **kwargs
    ):
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )

        # determinism
        self.random_state = kwargs.get('random_state')
        if self.random_state is not None:
            np.random.seed(self.random_state)
            set_seed(self.random_state)
        return

    # needed for advanced pipelines
    @property
    def _estimator_type(self):
        return "regressor"
