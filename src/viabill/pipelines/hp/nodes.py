from typing import Any, Dict, Tuple

import re
import warnings

import numpy as np
import pandas as pd

import scipy
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import category_encoders as ce
import logging

from functools import partial
from kedro.config import ConfigLoader
from hyperopt import fmin, hp, tpe
import lightgbm as lgb
import mlflow
from mlflow import log_metric, log_params
from mlflow.lightgbm import log_model
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.metrics import roc_auc_score

import category_encoders as ce

logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def _get_experiment() -> str:
    conf_paths = ["./conf/local", "./conf/base"]
    conf_loader = ConfigLoader(conf_paths=conf_paths)
    conf_mlflow = conf_loader.get("mlflow.yml")
    experiment_name = conf_mlflow\
        .get("experiment").get("name")
    client = MlflowClient(tracking_uri=conf_mlflow.get("tracking_uri"))
    experiments = client.list_experiments()
    lista = list(filter(lambda x: x.name == experiment_name, experiments))
    return lista[0].experiment_id if len(lista) > 0 else 0


def _get_tracking_uri() -> str:
    conf_paths = ["./conf/local", "./conf/base"]
    conf_loader = ConfigLoader(conf_paths=conf_paths)
    conf_mlflow = conf_loader.get("mlflow.yml")
    experiment_name = conf_mlflow\
        .get("experiment").get("name")
    return conf_mlflow.get("mlflow_tracking_uri")


def _objective(
        params: Dict,
        d_train: lgb.basic.Dataset,
        d_valid: lgb.basic.Dataset,
        tr_valid: np.ndarray,
        y_valid: np.ndarray,
        parameters) -> float:

    experiment_id = _get_experiment()

    mlflow.lightgbm.autolog(log_input_examples=False,
                            log_model_signatures=False,
                            log_models=True,
                            disable=False,
                            exclusive=False,
                            disable_for_unsupported_versions=False,
                            silent=False)

    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        params['deterministic'] = True
        params['objective'] = "binary"
        params['boosting'] = "gbdt"
        params['metric'] = ['auc', 'average_precision', 'binary_logloss']
        params['seed'] = '666'
        params['feature_pre_filter'] = False

        train_params = {
            'num_boost_round': parameters['num_boost_round'],
            'verbose_eval': parameters['verbose_eval'],
            'early_stopping_rounds': parameters['early_stopping_rounds'],
        }

        model = lgb.train(
            params,
            d_train,
            valid_sets=[d_train, d_valid],
            valid_names=['train', 'valid'],
            **train_params,
        )

        y_predict_valid = model.predict(tr_valid)

        auc = roc_auc_score(y_valid, y_predict_valid)

        return -1 * auc


def hp_tuning(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_valid: pd.DataFrame,
        parameters: Dict) -> Dict:

    space = {
        "learning_rate": hp.uniform("learning_rate", parameters["learning_rate"][0], parameters["learning_rate"][1]),
        #"max_bin": hp.randint("max_bin", parameters["max_bin"][0], parameters["max_bin"][1]),
        "max_depth": hp.randint("max_depth", parameters["max_depth"][0], parameters["max_depth"][1]),
        "min_data_in_leaf": hp.randint("min_data_in_leaf", parameters["min_data_in_leaf"][0], parameters["min_data_in_leaf"][1]),
        "num_leaves": hp.randint("num_leaves", parameters["num_leaves"][0], parameters["num_leaves"][1]),
        "lambda_l1": hp.uniform("lambda_l1", parameters["lambda_l1"][0], parameters["lambda_l1"][1]),
        "lambda_l2": hp.uniform("lambda_l2", parameters["lambda_l2"][0], parameters["lambda_l2"][1]),
        "bagging_fraction": hp.uniform("bagging_fraction", parameters["bagging_fraction"][0], parameters["bagging_fraction"][1]),
        "bagging_freq": hp.randint("bagging_freq", parameters["bagging_freq"][0], parameters["bagging_freq"][1]),
        "feature_fraction": hp.uniform("feature_fraction", parameters["feature_fraction"][0], parameters["feature_fraction"][1]),
        }

    # create holdout sets / valid

    categorical_cols = ['sex', 'residentialAddress_clean',
                        'postalAddress_clean', 'geo_risk_rank', 'shopID',
                        'same_address']
    numerical_cols = ['age', 'income', 'price'] + \
        df_train.filter(regex='(^hist_)|(_lst_)').columns.to_list()
    target_column = ['default']

    transformer = make_pipeline(
        ColumnTransformer([
            ('num', 'passthrough', numerical_cols),
            ('cat', ce.OrdinalEncoder(), categorical_cols),
        ]),
    )

    transformer.fit(df_train, df_train[parameters["target"]])

    d_train = lgb.Dataset(
        transformer.transform(df_train),
        label=df_train['default'],
        feature_name=transformer.named_steps['columntransformer'].get_feature_names(),
        params={'verbose': -1})

    d_valid = lgb.Dataset(
        transformer.transform(df_valid),
        label=df_valid[parameters["target"]],
        feature_name=transformer.named_steps['columntransformer'].get_feature_names(),
        params={'verbose': -1})

    tr_valid = transformer.transform(df_valid)
    y_valid = df_valid[parameters["target"]]

    best = fmin(
        partial(_objective,
                d_train=d_train,
                d_valid=d_valid,
                tr_valid=tr_valid,
                y_valid=y_valid,
                parameters=parameters),
        space,
        algo=tpe.suggest,
        max_evals=parameters["hp_number_of_experiments"])

    for k in best:
        if type(best[k]) != str:
            best[k] = str(best[k])

    return best
