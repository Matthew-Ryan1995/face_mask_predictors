import optuna
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import json

def objective(trial):
    x = pd.read_csv("data/X_train.csv", keep_default_na = False)
    y = pd.read_csv("data/y_train.csv", keep_default_na = False).values.ravel()

    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 300)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 200)

    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=rf_max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf

    )

    number_folds = 5
    kf = KFold(n_splits=number_folds)
    score = cross_val_score(rf, x, y, cv=kf, scoring='accuracy')
    accuracy = score.mean()
    trial.set_user_attr("std_err", np.std(score)/np.sqrt(number_folds))
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
        
    # Convert it into json file
    serialized_trials = []
    for trial in study.trials:
        serialized_trial = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }

    with open("data/rf_trials.json", "w") as outfile:
        for trial in study.trials:
                serialized_trial = {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "user_attrs": trial.user_attrs,
                }
                json.dump(serialized_trial, outfile)

    # find thr best one
    with open("data/rf_trial_best.json","w") as outfile:
        best_trial = study.best_trial
        serialized_trial = {
                        "number": best_trial.number,
                        "value": best_trial.value,
                        "params": best_trial.params,
                        "user_attrs": best_trial.user_attrs,
                    }
        
        json.dump(serialized_trial, outfile)