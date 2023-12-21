#!usr/bin/env python


# Internal libraries
from source.data_loaders import *
from source.dataset_builder import *
from source.ml.models import *

# External libraries
from sklearn.dummy import DummyRegressor


# Configure pandas to not truncate (...) row & column displays.
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


"""
# Download only last 2 years of data to to keep analysis focused on relevant.
download_transit_history(year="2023")
download_transit_history(year="2022")
"""

# Load a representative slice of historical bus-stop arrival time deviation data.
# This should be a fully continuous & resent slice to capture current behaviours,
# preserve time-series chronology & maintain route sequence structure of stops.
df = clean_transit_history(pd.concat((
    load_transit_history(frac=1.0, year="2023", month="05"),
    load_transit_history(frac=1.0, year="2023", month="06")
), ignore_index=True))

# Prepare for time-series learning from past to predict the future. Contextualize
# observations with previous stop & trip status & subsample to speedup learning.
df_learn, df_eval = dataset_splitter(df, column="Scheduled Time", test_frac=0.20, gap_frac=0.05)
df_learn = dataset_spatial_context(df_learn).sample(n=400000, replace=False)
df_eval = dataset_spatial_context(df_eval).sample(n=10000, replace=False)
del df

# Split input from output columns for test/train datasets. Goal is to predict
# current stop deviation given previous stop status & current stop characteristics.
X_learn = df_learn.drop("Deviation0", axis=1, inplace=False)
y_learn = df_learn["Deviation0"].to_frame()
X_eval = df_eval.drop("Deviation0", axis=1, inplace=False)
y_eval = df_eval["Deviation0"].to_frame()

# Define & train a collection of models to benchmark against.
model = {
    "dummy": DummyRegressor(strategy="mean"),     # random chance noise
    "base": build_base_model(spatial_window=5),   # baseline via classical ml
}
model = {
    name: model.fit(X_learn, y_learn)
    for name, model in model.items()
}

# Evaluate all models against the blind evaluation set & open learning-set
# to determine degree of noise, generalization & memorization.
print("Evaluation Set: (over-fitting?)")
print(report_evaluation(model, X_eval, y_eval))

print("Learning Set: (memorizing?)")
print(report_evaluation(model, X_learn, y_learn))

# TODO:
# -- implement trip history context window
# -- hypertune model parameters
# -- evaluate confidence intervals via cross-validation
# -- load & learn from larger time range
# -- efficiently random sample learn / eval sets
# -- enrich with road-network data at stop locations
# -- implement RNN equivalent model
# -- improve auto route detection via distance & timestamp


# EXPERIMENT 1: DATA BALANCING EFFECTS
# RESULTS:
# -- deteriorates performance.
# -- regression metric thrown off by artificial points
# ERROR       RUNTIME
# 5.473     	2.818       no sampling <-- best
# 7.108     	2.032       under sampling (3 cls)
# 6.681     	1.937       over sampling (3 cls)
# 6.053     	2.267       over sampling (2 cls)

# EXPERIMENT 2: BOOSTING vs BAGGING
# RESULTS:
# -- boosting is better
# -- sequential nature of problem might fit better
# ERROR       RUNTIME
# 5.473     	2.818       RandomForest (30 trees)
# 4.344     	1.556       XGBoost (30 trees) <-- best
# 4.904     	1.539       XGBoost (100 trees)
# 5.993     	1.84        XGBoost (1000 trees)

# EXPERIMENT 3: VECTOR REDUCTION
# RESULTS:
# -- deteriorates performance
# -- spatial & temporal dimensions alone are not compressible
# -- need more features!
# ERROR       RUNTIME
# 4.687     	1.499       no reduction at 1531 (100%) dims  <-- best
# 8.486     	1.742       TruncatedSVD at 1250 (~82%) dims
# 25.126    	1.73        TruncatedSVD at 100 (~66%) dims
# 9.243     	1.811       TruncatedSVD at 500 (~33%) dims

# EXPERIMENT 4: LEARNING DATA SIZE
# RESULTS:
# -- more data better performance!
# -- more stable than dummy
# -- training speed beyond 100k suffers
# ERROR       RUNTIME
# 5.294     	1.854       25k isolated ex
# 4.947     	2.32        40k isolated ex
# 3.927     	1.857       100k isolated ex
# 3.421     	1.991       200k isolated ex <-- best

# EXPERIMENT 5: SPATIAL CONTEXT WINDOW
# RESULTS:
# -- larger route stop window lookback the better
# -- diminishing returns beyond a point
# ERROR       RUNTIME
# 4.572     	1.804       0 stop window
# 3.057     	1.661       3 stop window
# 2.658     	1.896       5 stop window <-- best
