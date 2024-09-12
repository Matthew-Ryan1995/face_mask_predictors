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

# %%
model_number = "model_1a"
model_type = "rf"

with open(f"../models/{model_number}_{model_type}.pkl", "rb") as f:
    M = pickle.load(f)
f.close()

sort = M.feature_importances_.argsort()

X_train = pd.read_csv(
    f"../data/X_train_{model_number}.csv", keep_default_na=False)
y_train = pd.read_csv(
    f"../data/y_train_{model_number}.csv", keep_default_na=False)

# %%

rf_explainer = dx.Explainer(
    M, X_train, y_train, label="Face masks rf explainer")

# %%

pd_1 = rf_explainer.model_profile(
    variables=["age", "r1_1", "protective_behaviour_nomask_scale", "i2_health"])
pd_cat = rf_explainer.model_profile(
    variables=["within_mandate_period", "r1_1", "cantril_ladder"])

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

tmp = pd_1.plot(show=False)

tmp.write_image("test.png")
