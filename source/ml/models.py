#!usr/bin/env python

# Internal libraries
from source import RUNTIME_DIR
from source.ml.normalizers.sparsity import *
from source.ml.vectorizers.temporal import *
from source.ml.vectorizers.spatial import *
from source.ml.vectorizers.contextual import *
from source.ml.samplers.data_balancers import *

# External libraries
from joblib import wrap_non_picklable_objects
from imblearn.pipeline import Pipeline as ImbalancePipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline as ScikitPipeline, FeatureUnion
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklego.meta import EstimatorTransformer
from sklego.preprocessing import IdentityTransformer
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
    cache = False

    # Auto balance labels (only at fit time) by tolerance ranges.
    output_sampler = RegressionBalancer(
        sampling_mode="over",                   # clone minority
        fn_classifier=wrap_non_picklable_objects(
            lambda deviation:
                "late" if deviation < 0 else    # majority
                "early"                         # rare
        ),
        random_state=random_state,
    )

    # Capture cyclical date signal.
    temporal_vectorizer = DateTimeVectorizer(
        second_weight=0.0,      # too precise, conserve dimensions
        microsec_weight=0.0,    # too precise, conserve dimensions
    )

    # Quantize coverage areas by usage frequency
    spatial_vectorizer = GeoVectorizer(
        index_scheme='h3',  # hexagons have consistent areas
        items={'cells'},    # locations without relations
        resolution=8,       # 37/73% km² cell area of pentagons & hexagons
        binary=False,       # proxi bus schedule traffic & stop density
        max_items=3000,     # cap dimensionality to top most frequent
    )

    # Enrich observations with route & environment settings.
    contextual_vectorizer = RouteInfoVectorizer()

    # Enrich with anomaly signal treating sub-model as a transformer.
    anomaly_vectorizer = EstimatorTransformer(
        predict_func='predict',
        estimator=IsolationForest(
            n_estimators=100,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
    )

    # Feature select combining co-related ones
    dimensionality_reducer = PCA(
        n_components=300,
        random_state=random_state
    )
    # TruncatedSVD(
    #     n_components=1000,
    #     n_iter=10,
    #     n_oversamples=20,
    #     random_state=random_state,
    # )
    # <<dbg

    # Predict deviation time
    output_regressor = XGBRegressor(
        n_estimators=30,
        eval_metric=mean_squared_error,  # punish big errors more than small ones
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=int(verbose),
    )

    return ImbalancePipeline(
        steps=[
            ('balancer', output_sampler),
            ('vectorizer', ColumnTransformer(
                transformers=[
                    ('temporal', temporal_vectorizer, ["Scheduled Time"]),
                    ('spatial', spatial_vectorizer, ["Location"]),
                    ('contextual', contextual_vectorizer, ["Scheduled Time", "Route", "Stop Number"]),
                    ('historical', 'passthrough', [f"Deviation-{i}" for i in reversed(range(1, spatial_window + 1))])
                ],
                transformer_weights=None,
                remainder='drop',   # explicitly mask out all other columns
                verbose=verbose,
                n_jobs=n_jobs,
            )),
            # ('enricher', FeatureUnion(
            #     transformer_list=[
            #         #('identity', IdentityTransformer()),        # <<dbg looses get_feature_names_out() info!
            #         ('outliers', anomaly_vectorizer),            # <<dbg may need reshaping?
            #     ],
            #     transformer_weights=None,
            #     n_jobs=n_jobs,
            #     verbose=verbose,
            # )),
            ('denser', DenseTransformer()),
            ('scaler', RobustScaler(
                with_centering=True,    # destroy sparsity
                with_scaling=True,      # relative ranges
                unit_variance=False,    # preserve outliers
            )),
            ('reducer', dimensionality_reducer),
            ('estimator', output_regressor),
        ],
        memory=f'{RUNTIME_DIR}/cache' if cache else None,
        verbose=verbose,
    )


# Evaluates error & runtime of model inference against a given dataset.
def report_evaluation(models: dict, X, y) -> str:
    pad = 10
    precision = 3
    result = "\t".join([
        f"{'MODEL':<{pad}}",
        f"{'RMSE (min)':<{pad}}",
        f"{'R2 (%)':<{pad}}",
        f"RUNTIME (sec)\n"
    ])
    for name, model in models.items():
        t1 = time.time()
        y_actual = model.predict(X)
        t2 = time.time()
        metric_rmse = round(mean_squared_error(y_actual, y.to_numpy(), squared=False), precision)
        metric_r2 = round(r2_score(y_actual, y.to_numpy()) * 100, precision)  # should use adjusted r2
        metric_runtime = round(t2 - t1, precision)
        result += "\t".join([
            f"{name:<{pad}}",
            f"{metric_rmse:<{pad}}",
            f"{metric_r2:<{pad}}",
            f"{metric_runtime}\n"
        ])
    result += f"n={X.shape[0]}\n"
    return result
