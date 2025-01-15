#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:53:14 2024


@author: rya200
"""
# %%
import dalex as dx
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
model_number = "model_1a"
model_type = "rf"


def get_pdp_results(model_number, model_type):

    # read in model and training data
    with open(f"../models/{model_number}_{model_type}.pkl", "rb") as f:
        M = pickle.load(f)

    X_train = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y_train = pd.read_csv(
        f"../data/y_train_{model_number}.csv", keep_default_na=False)

    # Read in top features - calculated in R
    features = pd.read_csv(
        f"../results/{model_number}_{model_type}_top_features.csv")

    # Create the explainer
    rf_explainer = dx.Explainer(M,
                                X_train,
                                y_train,
                                label=f"{model_number}_{model_type}_explainer")

    # Explain the features
    pd_features = rf_explainer.model_profile(
        variables=features.features.to_list())

    # Save out results
    pd_features.result.to_csv(
        f"../results/{model_number}_{model_type}_pdp_results.csv")

# %%


model_numbers = ["model_" + a + b for a in ["1", "2"] for b in ["a", "b"]]
model_types = ["rf", "xgboost"]

for num in model_numbers:
    for tt in model_types:
        get_pdp_results(model_number=num, model_type=tt)

# pd_cat = rf_explainer.model_profile(
#     variables=["r1_1", "cantril_ladder", "d1_comorbidities_Yes",
#                "d1_comorbidities_No", "d1_comorbidities_Prefer_not_to_say"])

# %%
# tmp = pd_1.result[pd_1.result._vname_ == "age"]
# plt.figure()
# plt.plot(tmp._x_, tmp._yhat_)
# plt.show()

# tmp = pd_1.result[pd_1.result._vname_ == "protective_behaviour_nomask_scale"]
# plt.figure()
# plt.plot(tmp._x_, tmp._yhat_)
# plt.show()

# tmp = pd_cat.result[pd_cat.result._vname_ == "within_mandate_period"]
# plt.figure()
# plt.plot(tmp._x_, tmp._yhat_)
# plt.show()

# tmp = pd_cat.result[pd_cat.result._vname_ == "r1_1"]
# plt.figure()
# plt.plot(tmp._x_, tmp._yhat_)
# plt.show()

# tmp = pd_cat.result[pd_cat.result._vname_ == "cantril_ladder"]
# plt.figure()
# plt.plot(tmp._x_, tmp._yhat_)
# plt.show()

# tmp = pd_features.plot(show=False)

# tmp.write_image("test.png")

# tmp = pd_cat.plot(show=False)

# tmp.write_image("test2.png")
