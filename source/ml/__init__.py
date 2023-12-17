#!usr/bin/env python


# External libraries
from functools import partial, wraps
import numpy as np
import pandas as pd
import sklearn


# Decorator allowing scikit component fit/transform or predict/score methods to
# accept X inputs of various container types. Typically used as a convenience to
# avoid text input from having to explicitly be wrapped in np.array or when passing
# in dataframes.
def polymorphic_args(fn=None):
    if fn is None:
        return partial(polymorphic_args)

    @wraps(fn)
    def decorator(*args, **kwargs):
        x_raw = args[1]  # assumed (self, X, y, ..)
        input_type = type(x_raw)

        # convert polymorphic input to canonical
        x_transformed = \
            x_raw if input_type is np.ndarray else \
            x_raw.to_numpy() if input_type is pd.DataFrame else \
            np.array(x_raw) if input_type is list else \
            np.array([x_raw]) if input_type is str else \
            x_raw

        # invoke function with overwritten params
        args = list(args)
        args[1] = x_transformed
        y_raw = fn(*args, **kwargs)

        # convert result back to original input type
        return \
            np.array(y_raw) if input_type is np.ndarray else \
            pd.DataFrame(y_raw) if input_type is pd.DataFrame else \
            y_raw if input_type is list else \
            y_raw[0] if input_type is str else \
            y_raw

    # disable this decorator to reduce type converting overhead when scikit
    # checks are also disabled.
    return \
        fn if sklearn.get_config().get('assume_finite', False) else \
        decorator
