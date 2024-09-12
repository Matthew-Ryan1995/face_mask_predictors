#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:59:56 2024

@author: rya200
"""
# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%


def feature_importance_plot(model_number, model_type, number_of_features=10):

    save_file = f"../figures/feature_importance_{model_number}_{model_type}.png"

    with open(f"../models/{model_number}_{model_type}.pkl", "rb") as f:
        M = pickle.load(f)
    f.close()

    sort = M.feature_importances_.argsort()

    X_train = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)

    feature_importance = M.feature_importances_[sort]

    relative_feature_importance = (
        feature_importance/max(feature_importance)) * 100

    cols = X_train.columns[sort]

    plt.figure(figsize=(10, 6))
    plt.barh(cols[-number_of_features:],
             relative_feature_importance[-number_of_features:])
    plt.xlabel("Relative feature importance")
    plt.yticks(fontsize=15)
    plt.title(f"{model_number}: {model_type}")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()

# %%


model_numbers = ["model_1", "model_2", "model_1a", "model_2a"]
model_types = ["xgboost", "rf"]

for m_num in model_numbers:
    for m_type in model_types:
        feature_importance_plot(model_number=m_num, model_type=m_type)
