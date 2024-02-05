import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

cleaned_df = pd.read_csv("data/cleaned_data_preprocessing.csv", keep_default_na = False)
feature_cols = cleaned_df.columns.drop(["RecordNo", "face_mask_behaviour_scale", "face_mask_behaviour_binary", "endtime"])
for col in feature_cols:
    print(col)

# Label encoder
label_encoder = LabelEncoder()

for col in feature_cols:
    if cleaned_df[col].dtype == 'O':  # Checking if the column contains object (string) values
        cleaned_df[col] = label_encoder.fit_transform(cleaned_df[col])
    if cleaned_df[col].dtype == 'bool':
        cleaned_df[col] = cleaned_df[col].astype(int)

X = cleaned_df[feature_cols] # Features
y = label_encoder.fit_transform(cleaned_df.face_mask_behaviour_binary)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=21) # 10% test
# save these sets into csv file
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
pd.DataFrame({'y_train': y_train}).to_csv("data/y_train.csv", index=False)
pd.DataFrame({'y_test': y_test}).to_csv("data/y_test.csv", index=False)

print(X_train['within_mandate_period'].value_counts())