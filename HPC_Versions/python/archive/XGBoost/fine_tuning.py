import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def objective(trial):
    x = pd.read_csv("../data/X_train.csv", keep_default_na = False)
    y = pd.read_csv("../data/y_train.csv", keep_default_na = False).values.ravel()

    # learning_rate = trial.suggest_float("learning_rate", 0.01, 1)
    # max_depth = trial.suggest_int("max_depth", 2, 35)
    # min_child_weight = trial.suggest_int("min_child_weight", 1, 10) # 1- imbalance
    # subsample = trial.suggest_int("subsample", 0.1, 1)

    # colsample_bytree = trial.suggest_int("colsample_bytree", 0.5, 0.9)
    scale_pos_weight = trial.suggest_int("scale_pos_weight", 1, 10) # 1- imbalance
    # gamma = trial.suggest_int("gamma", 0, 5)
    # alpha = trial.suggest_float("alpha", 0.0, 2.0)
    # reg_lambda = trial.suggest_float("lambda", 0.0, 2.0)

    xgb_model = xgb.XGBClassifier(
                            # learning_rate=learning_rate,
                            # max_depth=max_depth,
                            # min_child_weight=min_child_weight,
                            # subsample=subsample,
                            # colsample_bytree=colsample_bytree,
                            scale_pos_weight=scale_pos_weight,
                            # gamma=gamma,
                            # alpha=alpha,
                            # reg_lambda=reg_lambda,
                            n_estimators = 1000,
                            objective="binary:logistic",
    )

    number_folds = 5
    kf = KFold(n_splits=number_folds)
    score = cross_val_score(xgb_model, x, y, cv=kf, scoring='accuracy')
    accuracy = score.mean()
    trial.set_user_attr("std_err", np.std(score)/np.sqrt(number_folds))
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=100, n_jobs=-1)
    serialized_trials = []
    for trial in study.trials:
        serialized_trial = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }

    with open("data/xgb_trials.json", "w") as outfile:
        for trial in study.trials:
                serialized_trial = {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "user_attrs": trial.user_attrs,
                }
                json.dump(serialized_trial, outfile)

    # find thr best one
    with open("data/xgb_trial_best.json","w") as outfile:
        best_trial = study.best_trial
        serialized_trial = {
                        "number": best_trial.number,
                        "value": best_trial.value,
                        "params": best_trial.params,
                        "user_attrs": best_trial.user_attrs,
                    }
        
        json.dump(serialized_trial, outfile)