#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:49:45 2024

Find important hyperparameters for optimizing roc_auc for model 1.

Findings: Consistently find that
    - max_depth
    - min_sample_leaf

### 230 seems best estimators.  Fix at 250

@author: rya200
"""
# %% Packages
import optuna
import pandas as pd
from sklearn.model_selection import cross_validate, KFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


# %% Function definitions
def objective(trial):
    # Define parameter ranges
    rf_max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    n_estimators_max = trial.suggest_int("n_estimators", 50, 1000)
    criterion_list = trial.suggest_categorical(
        'criterion', ['gini', 'entropy'])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 300)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 200)
    max_features = trial.suggest_categorical(
        'max_features', ['sqrt', 'log2', None])

    # Additional parameters
    class_weight = trial.suggest_categorical(
        'class_weight', [None, 'balanced', 'balanced_subsample'])
    # ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.5, step=0.01)
    # min_impurity_decrease = trial.suggest_float(
    #     "min_impurity_decrease", 0.0, 0.5, step=0.01)

    clf = RandomForestClassifier(
        n_estimators=n_estimators_max,
        max_depth=rf_max_depth,
        criterion=criterion_list,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,  # Set bootstrap to True
        # ccp_alpha=ccp_alpha,
        # min_impurity_decrease=min_impurity_decrease,
        class_weight=class_weight
    )

    n_splits = 5
    seed = 20240627
    kf = StratifiedShuffleSplit(n_splits=n_splits,
                                test_size=1/n_splits,
                                random_state=seed)
    score = cross_validate(clf, x, y, cv=kf, scoring=["roc_auc"])
    score = cross_validate(clf, x, y, cv=kf, scoring=["roc_auc"])
    roc = score["test_roc_auc"].mean()
    return roc


# %% Main
if __name__ == "__main__":
    start_time = datetime.now()
    # Load data
    model_number = "model_2"
    model_type = "rf"

    print(f'{model_type}-{model_number}')

    x = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv",
                    keep_default_na=False).values.ravel()

    # Run limited number of trails to see which are effecting results most
    study = optuna.create_study(directions=["maximize"],
                                sampler=optuna.samplers.TPESampler(seed=2013))
    study.optimize(objective, n_trials=250, n_jobs=mc_cores)
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(
        f"../figures/{model_number}_{model_type}_hyperparameter_importance.png")

    print(f"time taken: {datetime.now() - start_time}")
