import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier

def objective(trial):
    x = pd.read_csv("data/X_train.csv", keep_default_na = False)
    y = pd.read_csv("data/y_train.csv", keep_default_na = False).values.ravel()

    # Define parameter ranges
    param_ranges = {
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.2),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
    }

    # Create the DecisionTreeClassifier with suggested parameters
    clf = DecisionTreeClassifier(**param_ranges)
    kf = KFold(n_splits=5)
    score = cross_val_score(clf, x, y, cv=kf, scoring='accuracy')
    accuracy = score.mean()
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=100, n_jobs=-1)
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.show()
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    print(study.best_trial)

    # Convert it into json file
    serialized_trials = []
    for trial in study.trials:
        serialized_trial = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }

    with open("data/clf_trials.json", "w") as outfile:
        for trial in study.trials:
                serialized_trial = {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "user_attrs": trial.user_attrs,
                }
                json.dump(serialized_trial, outfile)

    # find thr best one
    with open("data/clf_trial_best.json","w") as outfile:
        best_trial = study.best_trial
        serialized_trial = {
                        "number": best_trial.number,
                        "value": best_trial.value,
                        "params": best_trial.params,
                        "user_attrs": best_trial.user_attrs,
                    }
        
        json.dump(serialized_trial, outfile)