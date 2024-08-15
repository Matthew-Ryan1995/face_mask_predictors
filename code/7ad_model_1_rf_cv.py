'''
Cross validation for model 1

Note to self: 
    Sensitivity = Recall, how many positives correctly predicted
    Precision: How many protected positives are correct
    
ToDo: Up-sampling on unbalanced data'
- max_depth
- min_samples_leaf
- max_features
- min_samples_split
Author:
    Jinjing Ye, Matt Ryan
    
Date created:
    17/04/2024
'''
# %% Packages
from datetime import datetime
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold, StratifiedShuffleSplit
import pickle

# %%

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
params["min_samples_split"] = int(params["min_samples_split"])


# %%

n_splits = 5
seed = 20240627
kf = StratifiedShuffleSplit(n_splits=n_splits,
                            test_size=1/n_splits,
                            random_state=seed)

metric_list = ['precision', "recall", "roc_auc", "accuracy", "f1"]

model = RandomForestClassifier(
    **params
)


# %%


def cross_validate_model(model_number):
    # Load data
    x = pd.read_csv(
        f"../data/X_train_{model_number}.csv", keep_default_na=False)
    y = pd.read_csv(f"../data/y_train_{model_number}.csv",
                    keep_default_na=False).values.ravel()

    # Cross validate model
    cv_scores = cross_validate(model, x, y, cv=kf, scoring=metric_list)

    # Print the accuracy scores for each fold

    print(f"{model_type}-{model_number}")
    # print("Cross-validation scores:", cv_scores)

    print("Mean recall: ", cv_scores["test_recall"].mean().round(3))
    print("Mean roc: ", cv_scores["test_roc_auc"].mean().round(3))
    print("Mean accuracy: ", cv_scores["test_accuracy"].mean().round(3))

    # Save results

    with open(f"../results/{model_number}_{model_type}.pkl", "wb") as f:
        pickle.dump(cv_scores, f)


# %%
start_time = datetime.now()
cross_validate_model(model_number)
print(f"time taken: {datetime.now() - start_time}")
