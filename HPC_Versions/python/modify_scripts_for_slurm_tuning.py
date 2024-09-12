#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 07:36:39 2024

@author: rya200
"""
# %% Organise job script titles

model_numbers = ["model_1", "model_2",
                 "model_1a", "model_2a",
                 "model_1b", "model_2b"]
model_letters = ["a", "b", "c", "d", "e", "f"]
model_types = ["binary_tree", "xgboost", "rf"]
model_numerals = ["5", "6", "7"]

model_combos = [
    f"{x}{y}b_{m_num}_{m_type}_tuning.py" for m_num, y in zip(model_numbers, model_letters) for m_type, x in zip(model_types, model_numerals)]

insert_chunk = "import os\ntry:\n    mc_cores = os.environ['SLURM_NTASKS']\n    mc_cores = int(mc_cores)\n    os.chdir(os.getcwd() + '/code')\nexcept:\n    mc_cores = 1\n\n"
print_script = "\n    print(f'{model_type}-{model_number}')\n\n"
for k in range(len(model_combos)):
    original_file = "../../code/" + model_combos[k]
    with open(original_file, 'r') as b:
        lines = b.readlines()
    with open(model_combos[k], 'w') as f:
        for i, line in enumerate(lines):
            # if line == '# %% Function definitions\n':
            #     f.write(insert_chunk)
            if "n_jobs=-1" in line:
                line = line.replace("n_jobs=-1", "n_jobs=mc_cores")
            # if "n_trials=50" in line:
            #     line = line.replace("n_trials=50", "n_trials=250")
            # if "n_trials=250" in line:
            #     line = line.replace("n_trials=250", "n_trials=10")
            if "fig.show()" in line:
                continue
            if "x = pd" in line:
                f.write(print_script)
            f.write(line)
