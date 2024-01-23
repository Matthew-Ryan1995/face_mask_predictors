import optuna
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import json
from sklearn.metrics import roc_auc_score

def objective(trial):
    cleaned_df = pd.read_csv("data/cleaned_data_preprocessing.csv", keep_default_na = False)

    # Feature Selection
    feature_cols = ['week_number', 'within_mandate_period', 'i2_health', 'i9_health', 'i11_health',
        'i12_health_2', 'i12_health_3', 'i12_health_4',
        'i12_health_5', 'i12_health_6', 'i12_health_7', 'i12_health_8',
        'i12_health_11', 'i12_health_12', 'i12_health_13', 'i12_health_14',
        'i12_health_15', 'i12_health_16','age', 
        'household_size', 'WCRex2', 'cantril_ladder', 
        'PHQ4_1', 'PHQ4_2', 'PHQ4_3', 'PHQ4_4',
        'd1_health_1', 'd1_health_2', 'd1_health_3', 'd1_health_4', 'd1_health_5', 'd1_health_6',
        'd1_health_7', 'd1_health_8', 'd1_health_9', 'd1_health_10',
        'd1_health_11', 'd1_health_12', 'd1_health_13', 'd1_health_98', 'd1_health_99',
        'WCRex1', 'r1_1', 'r1_2', 'state_New South Wales',
        'state_Northern Territory', 'state_Queensland', 'state_South Australia',
        'state_Tasmania', 'state_Victoria', 'state_Western Australia',
        'gender_Male', 'employment_status_Not working',
        'employment_status_Part time employment', 'employment_status_Retired',
        'employment_status_Unemployed']

    # Label encoder
    label_encoder = LabelEncoder()

    for col in feature_cols:
        if cleaned_df[col].dtype == 'O':  # Checking if the column contains object (string) values
            cleaned_df[col] = label_encoder.fit_transform(cleaned_df[col])

    x = cleaned_df[feature_cols] # Features
    y = cleaned_df.face_mask_behaviour_binary # Target variable

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