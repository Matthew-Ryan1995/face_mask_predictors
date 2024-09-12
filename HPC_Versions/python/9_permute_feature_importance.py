#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 08:30:01 2024

@author: rya200
"""
# %% libraries
import json
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import multiprocessing
from datetime import datetime
import os

start_time = datetime.now()

# %% Script parameters

num_perm = 10000


try:
    mc_cores = os.environ["SLURM_NTASKS"]
    mc_cores = int(mc_cores)
    os.chdir(os.getcwd() + "/code")
except:
    mc_cores = 1
try:
    array_val = os.environ["SLURM_ARRAY_TASK_ID"]
    array_val = int(array_val)-1
except:
    array_val = 0

print("Starting job ", array_val)

# %% functions


def fit_model_with_upsample(model_number, model_type, model):
    # Load data
    x = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv",
                    keep_default_na=False).values.ravel()

    upsampler = RandomOverSampler()

    X_train_upsample, y_train_upsample = upsampler.fit_resample(
        x, y)

    model_fitted = model.fit(X_train_upsample, y_train_upsample)
    # Cross validate model

    return model_fitted


def permute_feature_importance(perm_params):

    model_number, model_type, perm_number = perm_params
    print(f"Permutation {perm_number}")
    with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
        params = json.load(f)
    f.close()

    del params["number"]
    del params["value"]
    del params["std_err"]

    params["n_estimators"] = 250

    p_counter = 0
    try:
        params["max_depth"] = int(params["max_depth"])
    except:
        p_counter += 1
    try:
        params["min_samples_leaf"] = int(params["min_samples_leaf"])
    except:
        p_counter += 1
    try:
        params["min_samples_split"] = int(params["min_samples_split"])
    except:
        p_counter += 1
    try:
        params["min_child_weight"] = int(params["min_child_weight"])
    except:
        p_counter += 1

    if model_type == "xgboost":
        model = XGBClassifier(
            **params
        )
    else:
        model = RandomForestClassifier(
            **params
        )

    M = fit_model_with_upsample(model_number=model_number,
                                model_type=model_type, model=model)

    feature_importance = M.feature_importances_
    X_train = pd.read_csv(f"../data/X_train_{model_number}.csv",
                          keep_default_na=False)

    importance_dict = {}
    for name, value in zip(X_train.columns, feature_importance):
        importance_dict.update({name: value})

    return importance_dict


model_numbers = ["model_1", "model_2", "model_1a",
                 "model_2a", "model_1b", "model_2b"]
model_types = ["xgboost", "rf"]

model_combos = [(mn, mt) for mn in model_numbers for mt in model_types]


# %% Permute models for feature importance

if __name__ == '__main__':

    print(f"CORES: {mc_cores}")
    perm_params = [(x[0], x[1], i) for x in [model_combos[array_val]]
                   for i in range(num_perm)]

    print(f"{perm_params[0][1]} - {perm_params[0][0]}")
    save_file = f"../results/{perm_params[0][0]}_{perm_params[0][1]}_feature_importance.csv"
    if os.path.exists(save_file):
        print("Done")
    else:
        with multiprocessing.Pool(mc_cores) as p:
            ans = list(p.map(permute_feature_importance, perm_params))

        df = pd.DataFrame(ans)
        df.to_csv(save_file)

    print(f"Time taken: {datetime.now()-start_time}")


# %% Model 1a - XGBoost

# model_number = "model_1a"
# model_type = "xgboost"


# with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
#     params = json.load(f)
# f.close()

# del params["number"]
# del params["value"]
# del params["std_err"]


# params["n_estimators"] = 250


# model = XGBClassifier(
#     **params
# )

# M = fit_model_with_upsample(model_number=model_number,
#                         model_type=model_type, model=model)

# feature_importance = M.feature_importances_
# X_train = pd.read_csv(f"../data/X_train_{model_number}.csv",
#                       keep_default_na=False)

# importance_dict = {}
# for name, value in zip(X_train.columns, feature_importance):
#     importance_dict.update({name: value})


# # %% Model 1a - RF

# model_number = "model_1a"
# model_type = "rf"


# with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
#     params = json.load(f)
# f.close()

# del params["number"]
# del params["value"]
# del params["std_err"]


# params["n_estimators"] = 250
# params["max_depth"] = int(params["max_depth"])
# params["min_samples_leaf"] = int(params["min_samples_leaf"])
# params["min_samples_split"] = int(params["min_samples_split"])

# model = RandomForestClassifier(
#     **params
# )

# fit_model_with_upsample(model_number=model_number,
#                         model_type=model_type, model=model)

# %%
# model_number = model_numbers[0]
# model_type = model_types[0]

# save_file = f"../figures/feature_importance_{model_number}_{model_type}.png"

# with open(f"../models/{model_number}_{model_type}.pkl", "rb") as f:
#     M = pickle.load(f)


# sort = M.feature_importances_.argsort()
# feature_importance = M.feature_importances_
# X_train = pd.read_csv(
#     f"../data/X_train_{model_number}.csv", keep_default_na=False)

# importance_dict = {}
# for name, value in zip(X_train.columns, feature_importance):
#     importance_dict.update({name: value})
