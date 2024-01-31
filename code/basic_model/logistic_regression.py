import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import json
from sklearn.linear_model import LogisticRegression

x = pd.read_csv("../data/X_train.csv", keep_default_na = False)
y = pd.read_csv("../data/y_train.csv", keep_default_na = False).values.ravel()

# instantiate the model
logreg = LogisticRegression(random_state=16, max_iter = 5000)
kf = KFold(n_splits=10)
cv_scores = cross_val_score(logreg, x, y, cv=5, scoring='accuracy')

# Print the accuracy scores for each fold
print("Cross-validation scores:", cv_scores)

# Calculate and print the mean accuracy
print("Mean accuracy:", cv_scores.mean())