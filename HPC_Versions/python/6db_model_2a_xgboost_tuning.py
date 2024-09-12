#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:49:45 2024

Find important hyperparameters for optimizing roc_auc for model 1.

Findings: Consistently find that
    - learning rate
    - subsample
All useful for model fitting.  Tune these.

@author: rya200
"""
# %% Packages
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold, StratifiedShuffleSplit
import xgboost as xgb
from datetime import datetime
import json
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score


# %% Function definitions
def objective(trial):
    # Define parameter ranges
    learning_rate = trial.suggest_float("learning_rate", 0.01, 1)
    # max_depth = trial.suggest_int("max_depth", 2, 10)
    # min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    subsample = trial.suggest_float("subsample", 0.1, 1)

    # colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 0.9)
    # scale_pos_weight = trial.suggest_int("scale_pos_weight", 1, 10)
    scale_pos_weight = sum(1-y)/sum(y)
    # gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
    # alpha = trial.suggest_float("alpha", 0.0, 2.0)
    # reg_lambda = trial.suggest_float("lambda", 1e-8, 1.0, log=True)
    # n_estimator = trial.suggest_int("n_estimator", 100, 500)

    clf = xgb.XGBClassifier(
        learning_rate=learning_rate,
        # max_depth=max_depth,
        # min_child_weight=min_child_weight,
        subsample=subsample,
        # colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        # gamma=gamma,
        # alpha=alpha,
        # reg_lambda=reg_lambda,
        n_estimators=250,
        objective="binary:logistic",
    )
    number_folds = 5
    n_splits = 5
    seed = 20240627
    kf = StratifiedShuffleSplit(n_splits=n_splits,
                                test_size=1/n_splits,
                                random_state=seed)
    cv_scores = {
        "fold": [],
        'test_roc_auc': [],
    }

    splits = list(kf.split(x, y))

    for fold in range(len(splits)):
        cv_scores["fold"].append(fold)
        train_idx = splits[fold][0]
        val_idx = splits[fold][1]

        X_train = x.iloc[train_idx]
        y_train = y[train_idx]

        X_val = x.iloc[val_idx]
        y_val = y[val_idx]

        upsampler = RandomOverSampler()  # No seed set on purpose

        X_train_upsample, y_train_upsample = upsampler.fit_resample(
            X_train, y_train)

        clf_fitted = clf.fit(X_train_upsample, y_train_upsample)

        prop_preds = clf_fitted.predict_proba(X_val)

        cv_scores["test_roc_auc"].append(roc_auc_score(
            y_true=y_val, y_score=prop_preds[:, 1]))

    cv_scores["test_roc_auc"] = np.array(cv_scores["test_roc_auc"])

    roc = cv_scores["test_roc_auc"].mean()

    trial.set_user_attr("std_err",
                        np.std(cv_scores["test_roc_auc"])/np.sqrt(number_folds))

    return roc


# %% Main
if __name__ == "__main__":
    start_time = datetime.now()
    # Load data
    model_number = "model_2a"
    model_type = "xgboost"
    n_trials = 1000

    print(f'{model_type}-{model_number}')

    x = pd.read_csv(f"../data/X_train_{model_number}.csv",
                    keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv",
                    keep_default_na=False).values.ravel()

    # Run limited number of trails to see which are effecting results most
    study = optuna.create_study(directions=["maximize"],
                                sampler=optuna.samplers.TPESampler(seed=2020))
    study.optimize(objective, n_trials=n_trials, n_jobs=mc_cores)

    # Convert it into json file
    serialized_trials = []
    for trial in study.trials:
        serialized_trial = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }

    with open(f"../results/{model_number}_{model_type}_trials.json", "w") as outfile:
        for trial in study.trials:
            serialized_trial = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            json.dump(serialized_trial, outfile)
            outfile.write("\n")

    # find thr best one
    with open(f"../results/{model_number}_{model_type}_trial_best.json", "w") as outfile:
        best_trial = study.best_trial
        serialized_trial = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        }

        json.dump(serialized_trial, outfile)

    print(f"time taken: {datetime.now() - start_time}")
