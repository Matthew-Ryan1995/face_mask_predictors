#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:25:47 2024

Collect the cross validation results for model 1

@author: rya200
"""
# %%
import json
import pickle
import numpy as np
import pandas as pd

# %%

file_count = "03"

model_number = "model_1a"

model_types = ["logistic_reg",
               "binary_tree",
               "xgboost",
               "rf"]

# %%
final_df = pd.DataFrame()

for idx in range(len(model_types)):

    with open(f"../results/{model_number}_{model_types[idx]}.pkl", "rb") as f:
        M = pickle.load(f)
    f.close()

    model_dict = {
        "model_number": model_number,
        "model_type": model_types[idx],
        "precision": M["test_precision"].mean(),
        "precision_std": M["test_precision"].std()/(np.sqrt(M["test_precision"].size)),
        "recall": M["test_recall"].mean(),
        "recall_std": M["test_recall"].std()/(np.sqrt(M["test_recall"].size)),
        "roc_auc": M["test_roc_auc"].mean(),
        "roc_auc_std": M["test_roc_auc"].std()/(np.sqrt(M["test_roc_auc"].size)),
        "accuracy": M["test_accuracy"].mean(),
        "accuracy_std": M["test_accuracy"].std()/(np.sqrt(M["test_accuracy"].size)),
        "f1": M["test_f1"].mean(),
        "f1_std": M["test_f1"].std()/(np.sqrt(M["test_f1"].size))
    }

    model_df = pd.DataFrame(model_dict, index=[idx])

    final_df = pd.concat((final_df, model_df))

# %% Save

final_df.to_csv(f"../results/{file_count}_{model_number}_final_results.csv")
