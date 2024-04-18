'''
Split data into a testing and training split.

We have three different problems we are investigating:
    1. Predicting face mask usage
    2. Predicting protective behaviour
    3. Predicting early time indicators

We have a single test/train split, and derive different response/features for each of the modelling goals.

Note to self: 
    Sensitivity = Recall, how many positives correctly predicted
    Precision: How many protected positives are correct
    
ToDo: Up-sampling on unbalanced data
    
Author:
    Jinjing Ye, Matt Ryan
    
Date created:
    17/04/2024
'''
# %% Packages
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

# %% Parameter set up

# initiate the model
logreg = LogisticRegression(max_iter=5000)
kf = KFold(n_splits=5)

model_fitting = "logistic_reg"
metric_list = ['precision', "recall", "roc_auc", "accuracy", "f1"]

# %%


def cross_validate_model(model_number):
    # Load data
    x = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv",
                    keep_default_na=False).values.ravel()

    # Cross validate model
    cv_scores = cross_validate(logreg, x, y, cv=kf, scoring=metric_list)

    # Print the accuracy scores for each fold

    print(model_number)
    # print("Cross-validation scores:", cv_scores)

    print("Mean recall: ", cv_scores["test_recall"].mean().round(3))
    print("Mean roc: ", cv_scores["test_roc_auc"].mean().round(3))
    print("Mean accuracy: ", cv_scores["test_accuracy"].mean().round(3))

    # Save results

    with open(f"../results/{model_number}_{model_fitting}.pkl", "wb") as f:
        pickle.dump(cv_scores, f)

# %% Model 1:


model_number = "model_1"

cross_validate_model(model_number)

# %% Model 1a:

model_number = "model_1a"

cross_validate_model(model_number)

# %% Model 2:

model_number = "model_2"

cross_validate_model(model_number)

# %% Model 2a:

model_number = "model_2a"

cross_validate_model(model_number)

# %%
# How to load
# with open("../results/model_1a_logistic_reg.pkl", "rb") as f:
#     tmp = pickle.load(f)
