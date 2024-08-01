#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:12:16 2024

Findings: Consistently find that
    - max_features
    - min_sample_leafs
    - max_depth
    - min_samples_split

@author: rya200
"""

# %% Packages

import json
import pandas as pd

# %% Parameter set up

model_number = "model_2b"
model_type = "rf"

# %% Read data

obj_list = list()

with open(f'../results/{model_number}_{model_type}_trials.json', 'r') as f:
    for jsonObj in f:
        obj_dict = json.loads(jsonObj)
        obj_list.append(obj_dict)
f.close()


for d in obj_list:
    d.update(d.pop("params", {}))
    d.update(d.pop("user_attrs", {}))

# %% Convert to dataframe

tmp = pd.DataFrame(obj_list)

# %% Read best score

with open(f'../results/{model_number}_{model_type}_trial_best.json', 'r') as f:
    best = json.load(f)
f.close()

within_one_std_err = best["value"] - best["user_attrs"]["std_err"]

# %% Find most parsimonious within one std err

best_shots = tmp.loc[tmp["value"] > within_one_std_err]

# Order chosen based on hyperparamter importance
sort_params = [
    "max_features",
    "min_samples_leaf",
    "max_depth",
    "min_samples_split"
]

ans = best_shots.sort_values(sort_params).iloc[0]

with open(f"../results/{model_number}_{model_type}_best_within_one.json", "w") as f:
    json.dump(dict(ans), f, default=str)
f.close()
