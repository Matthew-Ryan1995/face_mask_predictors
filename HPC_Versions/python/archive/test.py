# import pandas as pd

# x = pd.read_csv("data/X_train.csv", index=False)
# y = pd.read_csv("data/y_train.csv", index=False)
# print(y.info())

import pickle


# %%
# How to load
with open("../results/model_1a_logistic_reg.pkl", "rb") as f:
    M1 = pickle.load(f)
with open("../results/model_1b_logistic_reg.pkl", "rb") as f:
    M2 = pickle.load(f)
