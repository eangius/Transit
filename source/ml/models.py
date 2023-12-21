#!usr/bin/env python

# Internal libraries
from source.ml.vectorizers.temporal import *
from source.ml.vectorizers.spatial import *
from source.ml.vectorizers.contextual import *
from source.ml.regressors.recurrent import *
from source.ml.samplers.data_balancers import *

# External libraries
from imblearn.pipeline import Pipeline as ImbalancePipeline
from sklearn.pipeline import Pipeline as ScikitPipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from functools import partial
from xgboost import XGBRegressor
import time


# Simple regressor predicting the next bus arrival time for a given stop of a
# directional route. In its basic form, model treats individual stops as isolated
# observations from the rest of the transit network. Sequence interdependence between
# previous stops & previous trips can be modeled respectively through the spatial &
# temporal context windows assumed to be encoded in the input. While some additions,
# updates & removal of routes & stops are tolerated at predict time (with diminishing
# accuracy) the model was tuned to learn characteristics of real locations & would
# need to be retrained periodically to capture structural or environmental changes.
# This model is useful to benchmark performance & rank feature importance.
def build_base_model(
    verbose: bool = True,
    spatial_window: int = 3,    # number of stops back to consider
    temporal_window: int = 3,   # number of trips back to consider
) -> ScikitPipeline:
    n_jobs = 1
    random_state = 42

    # Auto balance labels (only at fit time) by tolerance ranges.
    output_sampler = RegressionBalancer(
        sampling_mode="over",               # clone minority
        fn_classifier=lambda deviation:
            "late" if deviation < 0 else    # majority
            "early",                        # rare
        random_state=random_state,
    )

    # Capture cyclical date signal.
    temporal_vectorizer = DateTimeVectorizer(
        second_weight=0.0,      # too precise, conserve dimensions
        microsec_weight=0.0,    # too precise, conserve dimensions
    )

    # Quantize coverage areas by usage frequency
    spatial_vectorizer = ScikitPipeline(
        steps=[
            ('h3', GeoVectorizer(
                index_scheme='h3',  # hexagons have consistent areas
                items={'cells'},    # quantize locations without relations
                resolution=9,       # 5-10% of 1km^2 to compromise cell volume
                binary=False,       # proxi bus schedule traffic & stop density
                max_items=3000,     # cap dimensionality to top most frequent
            )),
        ],
        verbose=verbose,
    )

    # Enrich observations with route & environment settings.
    contextual_vectorizer = RouteInfoVectorizer()
    # <<dbg should this be pre-trained from all data (at risk of leakage)
    # & passed into the pipeline?

    # Feature select combining co-related ones
    dimensionality_reducer = 'passthrough'
    # TruncatedSVD(
    #     n_components=1000,
    #     n_iter=10,
    #     n_oversamples=20,
    #     random_state=random_state,
    # )
    # <<dbg

    # Sequentially predict from current input & previous output.
    output_regressor = ScikitPipeline(
        steps=[
            ('scaler', RobustScaler(
                with_centering=False,   # preserve sparsity
                with_scaling=True,      # relative ranges
                unit_variance=False,    # preserve outliers
            )),
            ('regressor', XGBRegressor(
                n_estimators=30,
                eval_metric=mean_squared_error,  # punish big errors more than small ones
                n_jobs=n_jobs,
                random_state=random_state,
                verbosity=int(verbose),
            )),
        ],
    )

    return ImbalancePipeline(
        steps=[
            ('balancer', output_sampler),
            ('vectorizer', ColumnTransformer(
                transformers=[
                    ("temporal", temporal_vectorizer, ["Scheduled Time"]),
                    ("spatial", spatial_vectorizer, ["Location"]),
                    ('contextual', contextual_vectorizer, ["Scheduled Time", "Route", "Stop Number"]),
                    ('historical', 'passthrough', [f"Deviation-{i}" for i in reversed(range(1, spatial_window + 1))])
                ],
                transformer_weights=None,
                remainder='drop',   # explicitly mask out all other columns
                verbose=verbose,
                n_jobs=n_jobs,
            )),
            ('reducer', dimensionality_reducer),
            ('estimator', output_regressor),
        ],
        verbose=verbose,
    )


# Evaluates error & runtime of model inference against a given dataset.
def report_evaluation(models: dict, X, y) -> str:
    metric = partial(mean_squared_error, squared=False)  # root mean squared error
    pad = 10
    result = f"{'MODEL':<{pad}}\t{'ERROR (min)':<{pad}}\tRUNTIME (sec)\n"
    for name, model in models.items():
        t1 = time.time()
        y_actual = model.predict(X)
        t2 = time.time()
        error = metric(y.to_numpy(), y_actual)
        result += f"{name:<{pad}}\t{round(error, 3):<{pad}}\t{round(t2 - t1, 3)}\n"
    result += f"n={X.shape[0]}\n"
    return result
