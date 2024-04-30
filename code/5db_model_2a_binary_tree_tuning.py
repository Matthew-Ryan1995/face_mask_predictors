#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:49:45 2024

Find important hyperparameters for optimizing roc_auc for model 1.

Findings: Consistently find that
    - min_imputiry_decrease
    - min_weight_fraction_leaf
All useful for model fitting.  Tune these.

@author: rya200
"""
# %% Packages
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import json
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
import numpy as np


# %% Function definitions
def objective(trial):
    # Define parameter ranges
    param_ranges = {
        # 'max_depth': trial.suggest_int('max_depth', 1, 20),
        # 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        # 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        # 'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        # 'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        # 'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.2),
        # 'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
    }

    # Create the DecisionTreeClassifier with suggested parameters
    clf = DecisionTreeClassifier(**param_ranges)

    number_folds = 5
    kf = KFold(n_splits=number_folds)

    cv_scores = {
        "fold": [],
        'test_roc_auc': [],
    }

    splits = list(kf.split(x))

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
    model_type = "binary_tree"
    n_trials = 1000
    x = pd.read_csv(f"../data/X_train_{model_number}.csv",
                    keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv",
                    keep_default_na=False).values.ravel()

    # Run limited number of trails to see which are effecting results most
    study = optuna.create_study(directions=["maximize"],
                                sampler=optuna.samplers.TPESampler(seed=2020))
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

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
