import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import confusion_matrix

# Read in json file with highest value
f = open('../data/xgb_trial_best.json', 'r')
obj = json.loads(f.read())

# create parameters variables
param_values = []
for parm in obj['params']:
    param_values.append(obj['params'][parm])

f.close()

X_train = pd.read_csv("../data/X_train.csv", keep_default_na = False)
y_train = pd.read_csv("../data/y_train.csv", keep_default_na = False).values.ravel()

xgb_model = xgb.XGBClassifier(
                        learning_rate=param_values[0],
                        # max_depth=max_depth,
                        min_child_weight=param_values[1],
                        # subsample=subsample,
                        # colsample_bytree=colsample_bytree,
                        scale_pos_weight=param_values[2],
                        # gamma=gamma,
                        # alpha=alpha,
                        # reg_lambda=reg_lambda,
                        n_estimators = 1000,
                        objective="binary:logistic",
)

X_test = pd.read_csv("../data/X_test.csv", keep_default_na = False)
y_test = pd.read_csv("../data/y_test.csv", keep_default_na = False)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

#  Confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred_xgb)
print(f"The confusion matrix is {confusion_matrix}")

# Print the accuracy scores for 10-folder cross validation
kf = KFold(n_splits=10)
score = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='accuracy')
print("Cross-validation scores:", score)
print("Mean accuracy:", score.mean())

# print cross validation ROC curve
# Split dataset into training set and test set
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Convert 'Yes' and 'No' labels to 1 and 0
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
plt.savefig("../figures/xgb_roc_curve.png")

# Explore the feature importance
sort = xgb_model.feature_importances_.argsort()
plt.figure(figsize=(10, 6))
plt.barh(X_train.columns[sort[-10:]], xgb_model.feature_importances_[sort[-10:]])
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.yticks(fontsize=7)
plt.savefig("../figures/xgb_feature_importance.png")

# visualize the xgboost tree
# plot_tree(xgb_model)
# plt.savefig("../figures/xgb_visual.png")