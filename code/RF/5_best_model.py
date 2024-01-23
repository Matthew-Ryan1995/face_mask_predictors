import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Read in json file with highest value
f = open('data/rf_trial_best.json', 'r')
obj = json.loads(f.read())

# create parameters variables
param_values = []
for parm in obj['params']:
    param_values.append(obj['params'][parm])

f.close()

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

X = cleaned_df[feature_cols] # Features
y = cleaned_df.face_mask_behaviour_binary # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 30% test

rf = RandomForestClassifier(
                            n_estimators=1000,
                            max_depth= param_values[0],
                            min_samples_split =param_values[1],
                            min_samples_leaf = param_values[2]
                            )

# Print the accuracy scores for 10-folder cross validation
kf = KFold(n_splits=10)
score = cross_val_score(rf, X, y, cv=kf, scoring='accuracy')
print("Cross-validation scores:", score)
print("Mean accuracy:", score.mean())

# print cross validation ROC curve
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 30% test
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Convert 'Yes' and 'No' labels to 1 and 0
y_true_binary = (y_test== 'Yes').astype(int)
print(f"roc score: {roc_auc_score(y_true_binary, y_pred_proba)}")

# Calculate ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true_binary, y_pred_proba)

# Calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()