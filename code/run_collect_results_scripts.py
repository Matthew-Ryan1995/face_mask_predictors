#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 06:55:06 2024

@author: rya200
"""
# %% Packages
import subprocess
import os

# %% OS and get slurm values
# os.chdir(os.getcwd() + "/code")

# try:
#     mc_cores = os.environ["SLURM_NTASKS"]
#     mc_cores = int(mc_cores)
# except:
#     mc_cores = 1
# try:
#     array_val = os.environ["SLURM_ARRAY_TASK_ID"]
#     array_val = int(array_val)
# except:
#     array_val = 1

# print("Starting job ", array_val)

# %% Organise job script titles

model_numbers = ["model_1", "model_2",
                 "model_1a", "model_2a",
                 "model_1b", "model_2b"]
model_letters = ["a", "b", "c", "d", "e", "f"]
# model_types = ["binary_tree", "xgboost", "rf"]
# model_numerals = ["5", "6", "7"]

model_combos = [
    f"8{y}_collect_results_{m_num}.py" for m_num, y in zip(model_numbers, model_letters)]

# filename = model_combos[array_val]

# %% Run job

for filename in model_combos:
    subprocess.run(["python3", filename])
