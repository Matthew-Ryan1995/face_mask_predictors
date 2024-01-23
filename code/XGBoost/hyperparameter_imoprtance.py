import optuna
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import xgboost as xgb

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

    learning_rate = trial.suggest_int("learning_rate", 0.05, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 35)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10) # 1- imbalance
    subsample = trial.suggest_int("subsample", 0.5, 0.9)
    colsample_bytree = trial.suggest_int("colsample_bytree", 0.5, 0.9)
    scale_pos_weight = trial.suggest_int("scale_pos_weight", 1, 10) # 1- imbalance
    gamma = trial.suggest_int("gamma", 0, 5)
    alpha = trial.suggest_float("alpha", 0.0, 2.0)
    reg_lambda = trial.suggest_float("lambda", 0.0, 2.0)

    xgb_model = xgb.XGBClassifier(
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            min_child_weight=min_child_weight,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            scale_pos_weight=scale_pos_weight,
                            gamma=gamma,
                            alpha=alpha,
                            reg_lambda=reg_lambda,
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
    study.optimize(objective,n_trials=10, n_jobs=-1)
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.show()
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    print(study.best_trial)