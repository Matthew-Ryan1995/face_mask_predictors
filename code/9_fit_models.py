#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:43:15 2024

Fit the XGBoost and the RF on the full training data as a "final" model.

@author: rya200
"""
# %%
import json
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import pickle
import pandas as pd


# %% functions

def fit_model_full_data(model_number, model_type, model):
    # Load data
    x = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv",
                    keep_default_na=False).values.ravel()

    # Cross validate model
    model_fitted = model.fit(x, y)

    # Save results

    with open(f"../models/{model_number}_{model_type}.pkl", "wb") as f:
        pickle.dump(model_fitted, f)
    f.close()


def fit_model_with_upsample(model_number, model_type, model):
    # Load data
    x = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv",
                    keep_default_na=False).values.ravel()

    upsampler = RandomOverSampler(random_state=2024)

    X_train_upsample, y_train_upsample = upsampler.fit_resample(
        x, y)

    model_fitted = model.fit(X_train_upsample, y_train_upsample)
    # Cross validate model

    # Save results

    with open(f"../models/{model_number}_{model_type}.pkl", "wb") as f:
        pickle.dump(model_fitted, f)
    f.close()


# %% Model 1 - XGBoost
model_number = "model_1"
model_type = "xgboost"


with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
    params = json.load(f)
f.close()

del params["number"]
del params["value"]
del params["std_err"]

params["n_estimators"] = 250

model = XGBClassifier(
    **params
)

fit_model_full_data(model_number=model_number,
                    model_type=model_type, model=model)

# %% Model 1 - RF

model_number = "model_1"
model_type = "rf"


with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
    params = json.load(f)
f.close()

del params["number"]
del params["value"]
del params["std_err"]

params["n_estimators"] = 250
params["max_depth"] = int(params["max_depth"])
params["min_samples_leaf"] = int(params["min_samples_leaf"])

model = RandomForestClassifier(
    **params
)

fit_model_full_data(model_number=model_number,
                    model_type=model_type, model=model)

# %% Model 2 - XGBoost

model_number = "model_2"
model_type = "xgboost"


with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
    params = json.load(f)
f.close()

del params["number"]
del params["value"]
del params["std_err"]

params["max_depth"] = int(params["max_depth"])

params["n_estimators"] = 250


model = XGBClassifier(
    **params
)

fit_model_full_data(model_number=model_number,
                    model_type=model_type, model=model)

# %% Model 2 - RF


model_number = "model_2"
model_type = "rf"


with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
    params = json.load(f)
f.close()

del params["number"]
del params["value"]
del params["std_err"]

params["n_estimators"] = 250
params["max_depth"] = int(params["max_depth"])
params["min_samples_leaf"] = int(params["min_samples_leaf"])

model = RandomForestClassifier(
    **params
)

fit_model_full_data(model_number=model_number,
                    model_type=model_type, model=model)

# %% Model 1a - XGBoost

model_number = "model_1a"
model_type = "xgboost"


with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
    params = json.load(f)
f.close()

del params["number"]
del params["value"]
del params["std_err"]


params["n_estimators"] = 250


model = XGBClassifier(
    **params
)

fit_model_with_upsample(model_number=model_number,
                        model_type=model_type, model=model)

# %% Model 1a - RF

model_number = "model_1a"
model_type = "rf"


with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
    params = json.load(f)
f.close()

del params["number"]
del params["value"]
del params["std_err"]


params["n_estimators"] = 250
params["max_depth"] = int(params["max_depth"])
params["min_samples_leaf"] = int(params["min_samples_leaf"])
params["min_samples_split"] = int(params["min_samples_split"])

model = RandomForestClassifier(
    **params
)

fit_model_with_upsample(model_number=model_number,
                        model_type=model_type, model=model)

# %% Model 2a - XGBoost


model_number = "model_2a"
model_type = "xgboost"


with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
    params = json.load(f)
f.close()

del params["number"]
del params["value"]
del params["std_err"]

params["n_estimators"] = 250


model = XGBClassifier(
    **params
)

fit_model_with_upsample(model_number=model_number,
                        model_type=model_type, model=model)

# %% Model 2a - RF


model_number = "model_2a"
model_type = "rf"


with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
    params = json.load(f)
f.close()

del params["number"]
del params["value"]
del params["std_err"]


params["n_estimators"] = 250
params["max_depth"] = int(params["max_depth"])
params["min_samples_leaf"] = int(params["min_samples_leaf"])
params["min_samples_split"] = int(params["min_samples_split"])

model = RandomForestClassifier(
    **params
)

fit_model_with_upsample(model_number=model_number,
                        model_type=model_type, model=model)
