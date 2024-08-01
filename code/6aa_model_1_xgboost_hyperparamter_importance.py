#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:49:45 2024

Find important hyperparameters for optimizing roc_auc for model 1.

Findings: Consistently find that
    - subsample
    - learning rate
    - colsample_by_tree
    - min_child_weight

### 230 seems best estimators.  Fix at 250

@author: rya200
"""
# %% Packages
import optuna
import pandas as pd
from sklearn.model_selection import cross_validate, KFold, StratifiedShuffleSplit
import xgboost as xgb
from datetime import datetime


# %% Function definitions
def objective(trial):
    # Define parameter ranges
    learning_rate = trial.suggest_float("learning_rate", 0.01, 1)
    max_depth = trial.suggest_int("max_depth", 2, 10)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    subsample = trial.suggest_float("subsample", 0.1, 1)

    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 0.9)
    # scale_pos_weight = trial.suggest_int("scale_pos_weight", 1, 10)
    scale_pos_weight = sum(1-y)/sum(y)
    gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
    # alpha = trial.suggest_float("alpha", 0.0, 2.0)
    reg_lambda = trial.suggest_float("lambda", 1e-8, 1.0, log=True)
    # n_estimator = trial.suggest_int("n_estimator", 100, 500)

    clf = xgb.XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        gamma=gamma,
        # alpha=alpha,
        reg_lambda=reg_lambda,
        n_estimators=250,
        objective="binary:logistic",
    )
    n_splits = 5
    seed = 20240627
    kf = StratifiedShuffleSplit(n_splits=n_splits,
                                test_size=1/n_splits,
                                random_state=seed)
    score = cross_validate(clf, x, y, cv=kf, scoring=["roc_auc"])
    roc = score["test_roc_auc"].mean()
    return roc


# %% Main
if __name__ == "__main__":
    start_time = datetime.now()
    # Load data
    model_number = "model_1"
    model_type = "xgboost"
    x = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv",
                    keep_default_na=False).values.ravel()

    # Run limited number of trails to see which are effecting results most
    study = optuna.create_study(directions=["maximize"],
                                sampler=optuna.samplers.TPESampler(seed=1985))
    study.optimize(objective, n_trials=50, n_jobs=-1)
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    fig.write_image(
        f"../figures/{model_number}_{model_type}_hyperparameter_importance.png")

    print(f"time taken: {datetime.now() - start_time}")
