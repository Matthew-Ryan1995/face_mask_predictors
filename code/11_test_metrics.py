#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:18:16 2024

@author: rya200
"""

# %%
import json
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score

# %%


model_numbers = ["model_1", "model_2", "model_1a",
                 "model_2a", "model_1b", "model_2b"]

model_types = ["xgboost",
               "rf"]

# %%
res_list = []
for model_number in model_numbers:
    x = pd.read_csv(
        f"../data/X_test_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_test_{model_number}.csv",
                    keep_default_na=False).values.ravel()
    for model_type in model_types:
        with open(f"../models/{model_number}_{model_type}.pkl", "rb") as f:
            M = pickle.load(f)

        test_metrics = {
            "model_number": model_number,
            'model_type': model_type,
        }

        preds = M.predict(x)
        prop_preds = M.predict_proba(x)

        test_metrics.update({"test_precision": precision_score(y_true=y,
                                                               y_pred=preds)})
        test_metrics.update(
            {"test_recall": recall_score(y_true=y, y_pred=preds)})
        test_metrics.update({"test_roc_auc": roc_auc_score(
            y_true=y, y_score=prop_preds[:, 1])})
        test_metrics.update(
            {"test_accuracy": accuracy_score(y_true=y, y_pred=preds)})
        test_metrics.update({"test_f1": f1_score(y_true=y, y_pred=preds)})

        res_list.append(test_metrics)

final_df = pd.DataFrame(res_list)

final_df.to_csv("../results/final_model_test_metrics.csv")
