import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Read in json file with highest value
f = open('../data/rf_trial_best.json', 'r')
obj = json.loads(f.read())

# create parameters variables
param_values = []
for parm in obj['params']:
    param_values.append(obj['params'][parm])

f.close()

X_train = pd.read_csv("../data/X_train.csv", keep_default_na = False)
y_train = pd.read_csv("../data/y_train.csv", keep_default_na = False).values.ravel()

rf = RandomForestClassifier(
                            n_estimators=1000,
                            ccp_alpha= param_values[0],
                            min_impurity_decrease =param_values[1]
                            )

X_test = pd.read_csv("../data/X_test.csv", keep_default_na = False)
y_test = pd.read_csv("../data/y_test.csv", keep_default_na = False).values.ravel()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#  Confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred_rf)
print(f"The confusion matrix is {confusion_matrix}")

# Print the accuracy scores for 10-folder cross validation
kf = KFold(n_splits=10)
score = cross_val_score(rf, X_train, y_train, cv=kf, scoring='accuracy')
print("Cross-validation scores:", score)
print("Mean accuracy:", score.mean())

# print cross validation ROC curve
# Split dataset into training set and test set
y_pred_proba = rf.predict_proba(X_test)[:, 1]
print(f"roc score: {roc_auc_score(y_test, y_pred_proba)}")

# Calculate ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba)

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
plt.savefig("../figures/rf_roc_curve.png")

# Explore the feature importance
sort = rf.feature_importances_.argsort()
plt.figure(figsize=(10, 6))
plt.barh(X_train.columns[sort], rf.feature_importances_[sort])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.yticks(fontsize=6, rotation=45)
plt.savefig("../figures/rf_feature_importance.png")