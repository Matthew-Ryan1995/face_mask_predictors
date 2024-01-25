import pandas as pd

x = pd.read_csv("data/X_train.csv", index=False)
y = pd.read_csv("data/y_train.csv", index=False)
print(y.info())