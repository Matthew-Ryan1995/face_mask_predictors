import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from mlxtend.evaluate import bias_variance_decomp

cleaned_df = pd.read_csv("../data/cleaned_data_preprocessing.csv", keep_default_na = False)

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

X = cleaned_df[feature_cols] # Features
y = cleaned_df.face_mask_behaviour_binary # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 30% test

rf = RandomForestClassifier(
                            n_estimators=351, #random_state= 1,
                            max_depth = 20,                            
                            # min_samples_leaf= 1, 
                            )
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# kf = KFold(n_splits=3)
score = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

# Print the accuracy scores for each fold
print("Cross-validation scores:", score)
print("Mean accuracy:", score.mean())

# label_encoder = LabelEncoder()
# y_train_binary = label_encoder.fit_transform(y_train)
# y_test_binary = label_encoder.transform(y_test)

# print(metrics.accuracy_score(y_test, y_pred_rf))
# avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
#         rf, X_train.values, y_train_binary, X_test.values, y_test_binary, 
#         loss='mse',
#         random_seed=1)

# print(f'Average Expected Loss: {round(avg_expected_loss, 4)}')
# print(f'Average Bias: {round(avg_bias, 4)}')
# print(f'Average Variance: {round(avg_var, 4)}')