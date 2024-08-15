'''
Cross validation for model 1

Note to self: 
    Sensitivity = Recall, how many positives correctly predicted
    Precision: How many protected positives are correct
    
ToDo: Up-sampling on unbalanced data
        - min_imputiry_decrease
        - min_weight_fraction_leaf
Author:
    Jinjing Ye, Matt Ryan
    
Date created:
    17/04/2024
'''
# %% Packages
import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, KFold, StratifiedShuffleSplit
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score
import numpy as np

# %%

model_number = "model_2b"
model_type = "binary_tree"


with open(f'../results/{model_number}_{model_type}_best_within_one.json', 'r') as f:
    params = json.load(f)
f.close()

del params["number"]
del params["value"]
del params["std_err"]


# %%

n_splits = 5
seed = 20240627
kf = StratifiedShuffleSplit(n_splits=n_splits,
                            test_size=1/n_splits,
                            random_state=seed)

metric_list = ['precision', "recall", "roc_auc", "accuracy", "f1"]

model = DecisionTreeClassifier(
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
    cv_scores = {
        "fold": [],
        'test_precision': [],
        'test_recall': [],
        'test_roc_auc': [],
        'test_accuracy': [],
        'test_f1': []
    }

    splits = list(kf.split(x, y))

    for fold in range(len(splits)):
        cv_scores["fold"].append(fold)
        train_idx = splits[fold][0]
        val_idx = splits[fold][1]

        X_train = x.iloc[train_idx]
        y_train = y[train_idx]

        X_val = x.iloc[val_idx]
        y_val = y[val_idx]

        upsampler = RandomOverSampler()  # No seed set on purpose

        X_train_upsample, y_train_upsample = upsampler.fit_resample(
            X_train, y_train)

        clf_fitted = model.fit(X_train_upsample, y_train_upsample)

        preds = clf_fitted.predict(X_val)
        prop_preds = clf_fitted.predict_proba(X_val)

        cv_scores["test_precision"].append(
            precision_score(y_true=y_val, y_pred=preds))
        cv_scores["test_recall"].append(
            recall_score(y_true=y_val, y_pred=preds))
        cv_scores["test_roc_auc"].append(roc_auc_score(
            y_true=y_val, y_score=prop_preds[:, 1]))
        cv_scores["test_accuracy"].append(
            accuracy_score(y_true=y_val, y_pred=preds))
        cv_scores["test_f1"].append(f1_score(y_true=y_val, y_pred=preds))

    cv_scores["test_precision"] = np.array(cv_scores["test_precision"])
    cv_scores["test_recall"] = np.array(cv_scores["test_recall"])
    cv_scores["test_roc_auc"] = np.array(cv_scores["test_roc_auc"])
    cv_scores["test_accuracy"] = np.array(cv_scores["test_accuracy"])
    cv_scores["test_f1"] = np.array(cv_scores["test_f1"])

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


cross_validate_model(model_number)
