import pandas as pd
import numpy as np
import logging
import os
from src.common_paths import *
from src.logging_tools import setup_logging_environment
from src.file_loaders import load_numerai_data
from src.numerai_utilities import download_last_numerai_data
from src.reporting_tools import generate_correlation_matrices, generate_profiling_reports
from src.model_helpers import train_model_with_gridsearch
from src.model_battery import *
from src.numerai_utilities import build_submission, upload_submission
__author__ = "ivallesp"


version = "last"

setup_logging_environment()

# Download Data
if not os.listdir(get_raw_data_version_path(version)):
    status = download_last_numerai_data(version_name=version)
    generate_profiling_reports()
    generate_correlation_matrices()

# Prepare data
df_train, df_test = load_numerai_data(version)
df_whole = df_train.copy()
df_dev = df_train.sample(frac=0.1, random_state=655321)
df_train = df_train.drop(df_dev.index)

train_vars = df_train.columns[~df_train.columns.str.contains("target")]
target_var = "target"

# Launch Models
scores_dev = {}
results_json = {}

aliases = []
models = []
param_grids = []


aliases.append("NBGeneral")
model, params = get_general_nb()
models.append(model)
param_grids.append(params)

aliases.append("GLMNETGeneral")
model, params = get_general_glmnet()
models.append(model)
param_grids.append(params)

aliases.append("TREEGeneral")
model, params = get_general_tree()
models.append(model)
param_grids.append(params)

aliases.append("ETGeneral")
model, params = get_general_et()
models.append(model)
param_grids.append(params)

aliases.append("GLMGeneral")
model, params = get_general_glm()
models.append(model)
param_grids.append(params)

aliases.append("MLPGeneral")
model, params = get_general_mlp()
models.append(model)
param_grids.append(params)

aliases.append("KNNGeneral")
model, params = get_general_knn()
models.append(model)
param_grids.append(params)

aliases.append("MLPGeneral")
model, params = get_general_svc()
models.append(model)
param_grids.append(params)

aliases.append("RFGeneral")
model, params = get_general_rf()
models.append(model)
param_grids.append(params)


for alias, model, params in zip(aliases, models, param_grids):
    print "TRAIN MODEL WITH ALIAS %s"% alias
    model, results = train_model_with_gridsearch(X=df_train[train_vars],
                                                 y=df_train[target_var],
                                                 estimator=model,
                                                 param_grid=params,
                                                 n_jobs=1,
                                                 verbose=1)
    dev_score = -model.score(df_dev[train_vars], df_dev[target_var])
    model.fit(df_whole[train_vars],
              df_whole[target_var])
    preds = model.predict_proba(df_test[train_vars])[:,1]
    build_submission(version=version, indices=df_test.t_id, probs=preds, alias=alias)
    status, score = upload_submission(version = version, alias=alias)
    scores_dev["alias"] = dev_score
    results_json["alias"] = results

