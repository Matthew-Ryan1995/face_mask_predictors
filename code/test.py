import optuna
from sklearn import preprocessing
import pandas as pd
from datetime import datetime

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
from sklearn.preprocessing import LabelEncoder

def mandates_convert(index):
    endtime = cleaned_df["endtime"][index]
    state = df["state"][index]
    
    if states_date[state][0] <= endtime <= states_date[state][1]:
        return 1
    else:
        return 0
    
def objective(trial):
    cleaned_df = pd.read_csv("../data/cleaned_data.csv", keep_default_na = False)
    
    # add in one column  with face mask mandates
    states_date = {'Australian Capital Territory': ['2021-06-28', '2022-02-25'], 'New South Wales': ['2021-01-04', '2022-09-20'], 'Northern Territory': ['2021-12-19', '2022-03-05'], 'Queensland': ['2021-06-29', '2022-03-07'], 'South Australia': ['2021-07-27', '2022-09-20'], 'Tasmania': ['2021-12-21', '2022-03-05'], 'Victoria': ['2020-07-23', '2022-09-22'], 'Western Australia': ['2021-12-23', '2022-04-29']}
        
    for state, date_range in states_date.items():
        states_date[state] = [pd.to_datetime(date, format='%Y-%m-%d') for date in date_range]

    # Create a new column "period"
    cleaned_df['within_mandate_period'] = cleaned_df.index.map(lambda idx: mandates_convert(idx) if idx in cleaned_df.index else 0)

    # Create dummy variables for state, gender, employment_status
    state_dummies = pd.get_dummies(cleaned_df['state'], prefix='state', drop_first=True)

    # Create dummy variables for 'gender'
    gender_dummies = pd.get_dummies(cleaned_df['gender'], prefix='gender', drop_first=True)

    # Create dummy variables for 'employment_status'
    employment_status_dummies = pd.get_dummies(cleaned_df['employment_status'], prefix='employment_status', drop_first=True)

    # Concatenate the dummy variables with the original DataFrame
    cleaned_df = pd.concat([cleaned_df, state_dummies, gender_dummies, employment_status_dummies], axis=1)

    # Drop the original categorical columns
    cleaned_df = cleaned_df.drop(['state', 'gender', 'employment_status'], axis=1)

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

    # Enable labelEncoder()
    label_encoder = LabelEncoder()

    for col in feature_cols:
        if cleaned_df[col].dtype == 'O':  # Checking if the column contains object (string) values
            cleaned_df[col] = label_encoder.fit_transform(cleaned_df[col])
            
    x = cleaned_df[feature_cols] # Features
    y = cleaned_df.face_mask_behaviour_binary # Target variable
    rf_max_depth = trail.suggest_int("rf_max_depth", 2, 32, log=True)
    classifier_obj = sklearn.ensemble.RandomForestClassfier(
        max_depth=rf_max_depth, n_estimators=10
    )
    score = sklearn.model_selection.cross_val_score(classfier_obj, x, y, n_jobs=-1, cv=5)
    accuracy = score.mean()
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)