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
from sklearn.multioutput import RegressorChain
from functools import partial
from xgboost import XGBRegressor
import time


# ABOUT: simple bagging regressor treating individual stops independent
# from the route & transit network. Useful to benchmark performance &
# rank importance.
def build_base_model(verbose: bool = True) -> ScikitPipeline:
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

    # Capture cyclical date signal
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
            #dbg>> ('scaler', MaxAbsScaler(copy=False)),  # scale ranges preserving sparsity
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
            ('regressor', RegressorChain(
                base_estimator=XGBRegressor(
                    n_estimators=30,
                    eval_metric=mean_squared_error,  # punish big errors more than small ones
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbosity=int(verbose),
                ),
                order=[0],     # <<dbg single output for now
                random_state=random_state,
                verbose=verbose,
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
                ],
                transformer_weights=None,
                remainder='drop',   # ignore other columns
                verbose=verbose,
                n_jobs=n_jobs,
            )),
            ('reducer', dimensionality_reducer),
            ('estimator', output_regressor),
        ],
        verbose=verbose,
    )


def build_real_model(verbose: bool = True) -> ScikitPipeline:
    return ScikitPipeline(
        verbose=verbose,
        steps=[
            ('spatial', LSTMRegressor(
                bidirectional=False,    # avoid leakage
                mask_value=-1,          # flag padding
                units=10,               # hyper-param
                random_state=42,
                verbose=verbose,
            ))
        ]
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
