'''
Split data into a testing and training split.

We have three different problems we are investigating:
    1. Predicting face mask usage
    2. Predicting protective behaviour
    3. Predicting early time indicators

We have a single test/train split, and derive different response/features for each of the modelling goals.
    
Author:
    Jinjing Ye, Matt Ryan
    
Date created:
    17/04/2024
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

cleaned_df = pd.read_csv(
    "../data/cleaned_data_preprocessing.csv", keep_default_na=False)
# feature_cols = cleaned_df.columns.drop(
#     ["RecordNo", "face_mask_behaviour_scale", "face_mask_behaviour_binary", "endtime"])
# for col in feature_cols:
#     print(col)

# Label encoder
# label_encoder = LabelEncoder()
# %%

df_train, df_test = train_test_split(
    cleaned_df, test_size=0.1, random_state=20240417)

df_train.to_csv("../data/df_train.csv", index=False)
df_test.to_csv("../data/df_test.csv", index=False)

# %% Model 1: Predicting face masks

label_encoder = LabelEncoder()

response_col = ["face_mask_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "endtime"])

X_train_model_1 = df_train[feature_cols]
X_test_model_1 = df_test[feature_cols]

y_train_model_1 = label_encoder.fit_transform(df_train[response_col])
y_test_model_1 = label_encoder.fit_transform(df_test[response_col])

X_train_model_1.to_csv("../data/X_train_model_1.csv", index=False)
X_test_model_1.to_csv("../data/X_test_model_1.csv", index=False)
pd.DataFrame({'y_train': y_train_model_1}).to_csv(
    "../data/y_train_model_1.csv", index=False)
pd.DataFrame({'y_test': y_test_model_1}).to_csv(
    "../data/y_test_model_1.csv", index=False)

# %% Model 1a: Predicting face masks in early time

mandate_starter = '2020-07-23'


response_col = ["face_mask_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "endtime"])

X_train_model_1a = df_train[df_train["endtime"] <
                            mandate_starter][feature_cols]
X_test_model_1a = df_test[df_test["endtime"] < mandate_starter][feature_cols]

y_train_model_1a = label_encoder.fit_transform(
    df_train[df_train["endtime"] < mandate_starter][response_col])
y_test_model_1a = label_encoder.fit_transform(
    df_test[df_test["endtime"] < mandate_starter][response_col])

X_train_model_1a.to_csv("../data/X_train_model_1a.csv", index=False)
X_test_model_1a.to_csv("../data/X_test_model_1a.csv", index=False)
pd.DataFrame({'y_train': y_train_model_1a}).to_csv(
    "../data/y_train_model_1a.csv", index=False)
pd.DataFrame({'y_test': y_test_model_1a}).to_csv(
    "../data/y_test_model_1a.csv", index=False)


# %% Model 2: Predicting protective behaviour

label_encoder = LabelEncoder()

response_col = ["protective_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "protective_behaviour_nomask_scale",
                                        "endtime"])

X_train_model_2 = df_train[feature_cols]
X_test_model_2 = df_test[feature_cols]

y_train_model_2 = label_encoder.fit_transform(df_train[response_col])
y_test_model_2 = label_encoder.fit_transform(df_test[response_col])

X_train_model_2.to_csv("../data/X_train_model_2.csv", index=False)
X_test_model_2.to_csv("../data/X_test_model_2.csv", index=False)
pd.DataFrame({'y_train': y_train_model_2}).to_csv(
    "../data/y_train_model_2.csv", index=False)
pd.DataFrame({'y_test': y_test_model_2}).to_csv(
    "../data/y_test_model_2.csv", index=False)

# %% Model 2a: Predicting protective behaviour in early time

mandate_starter = '2020-07-23'


response_col = ["protective_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "protective_behaviour_nomask_scale",
                                        "endtime"])

X_train_model_2a = df_train[df_train["endtime"] <
                            mandate_starter][feature_cols]
X_test_model_2a = df_test[df_test["endtime"] < mandate_starter][feature_cols]

y_train_model_2a = label_encoder.fit_transform(
    df_train[df_train["endtime"] < mandate_starter][response_col])
y_test_model_2a = label_encoder.fit_transform(
    df_test[df_test["endtime"] < mandate_starter][response_col])

X_train_model_2a.to_csv("../data/X_train_model_2a.csv", index=False)
X_test_model_2a.to_csv("../data/X_test_model_2a.csv", index=False)
pd.DataFrame({'y_train': y_train_model_1a}).to_csv(
    "../data/y_train_model_2a.csv", index=False)
pd.DataFrame({'y_test': y_test_model_1a}).to_csv(
    "../data/y_test_model_2a.csv", index=False)


# for col in feature_cols:
#     # Checking if the column contains object (string) values
#     if cleaned_df[col].dtype == 'O':
#         cleaned_df[col] = label_encoder.fit_transform(cleaned_df[col])
#     if cleaned_df[col].dtype == 'bool':
#         cleaned_df[col] = cleaned_df[col].astype(int)

# X = cleaned_df[feature_cols]  # Features
# y = label_encoder.fit_transform(cleaned_df.face_mask_behaviour_binary)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, random_state=21)  # 10% test
# # save these sets into csv file
# X_train.to_csv("data/X_train.csv", index=False)
# X_test.to_csv("data/X_test.csv", index=False)
# pd.DataFrame({'y_train': y_train}).to_csv("data/y_train.csv", index=False)
# pd.DataFrame({'y_test': y_test}).to_csv("data/y_test.csv", index=False)

# print(X_train['within_mandate_period'].value_counts())
