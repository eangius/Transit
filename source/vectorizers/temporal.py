#!usr/bin/env python


# External libraries
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from typing import Tuple, Callable
import numpy as np
import time


class DateTimeVectorizer(FeatureUnion):
    """
    Trigonometrically encodes cyclical date-time attributes proportional to each of
    their weights. This reduces the curse of high dimensionality from one-hot-encoding
    each of the parts. When weights are all zero, encoding is just a scaler denoting
    seconds since unix epoch.
    """

    def __init__(
        self,
        month_weight: float = 1.0,
        weekday_weight: float = 1.0,
        hour_weight: float = 1.0,
        minute_weight: float = 1.0,
        second_weight: float = 1.0,
        microsec_weight: float = 1.0,
        **kwargs
    ):
        
        # store for persistence
        self.month_weight = month_weight
        self.weekday_weight = weekday_weight
        self.hour_weight = hour_weight
        self.minute_weight = minute_weight
        self.second_weight = second_weight
        self.microsec_weight = microsec_weight

        transformer_list = []
        transformer_weights = dict()

        if all(
            w == 0
            for w in {
                month_weight,
                weekday_weight,
                hour_weight,
                second_weight,
                minute_weight,
            }
        ):
            transformer_list.append(("time_abs", self._abs_feature()))
            transformer_weights = dict()
        else:
            for weights, feats in [
                self._build_feature("month", month_weight, 12, lambda dt: dt.month),
                self._build_feature("weekday", weekday_weight, 7, lambda dt: dt.weekday()),
                self._build_feature("hour", hour_weight, 24, lambda dt: dt.hour),
                self._build_feature("minute", minute_weight, 60, lambda dt: dt.minute),
                self._build_feature("second", second_weight, 60, lambda dt: dt.second),
                self._build_feature("microsecond", microsec_weight, 1000000, lambda dt: dt.microsecond),
            ]:
                transformer_list.extend(feats)
                transformer_weights.update(weights)

        super().__init__(
            transformer_list=transformer_list,
            transformer_weights=transformer_weights,
            n_jobs=kwargs.get("n_jobs"),
            verbose=kwargs.get("verbose", False),
        )

    @staticmethod
    def _build_feature(lbl: str, weight: float, period: int, fn: Callable) -> Tuple[dict, list]:
        weights = dict()
        features = []
        if weight != 0:
            weights = {
                f"{lbl}_sin": weight,
                f"{lbl}_cos": weight,
            }
            features = [
                (f"{lbl}_sin", DateTimeVectorizer._sin_feature(period, fn)),
                (f"{lbl}_cos", DateTimeVectorizer._cos_feature(period, fn)),
            ]
        return weights, features

    @staticmethod
    def _sin_feature(period: int, fn: Callable):
        return FunctionTransformer(
            lambda X: np.sin(np.array(list(map(fn, X))) / period * 2 * np.pi),
            check_inverse=False,
        )

    @staticmethod
    def _cos_feature(period: int, fn: Callable):
        return FunctionTransformer(
            lambda X: np.cos(np.array(list(map(fn, X))) / period * 2 * np.pi)
        )

    @staticmethod
    def _abs_feature():
        return FunctionTransformer(
            lambda X: np.array([time.mktime(x.timetuple()) for x in X])
        )
