# import pandas as pd
# import matplotlib.pyplot as plt
# import csv
# first_line = True

# # perform the same plot in UK
# df = pd.read_csv("raw_data/australia_a.csv")

# frequency_dict = {"Always":5, "Frequently":4, "Sometimes":3, "Rarely":2, "Not at all":1}
# mask = df["i12_health_1"]

# mask = mask.map(frequency_dict)
# print("i12_health_1: No. of N/A is", mask.isnull().sum(), "and the mean is", round(mask.mean(),2))
# print("\n")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("raw_data/united-kingdom.csv", encoding='ISO-8859-1')  

# Frequency dictionary
frequency_dict = {"Always": 5, "Frequently": 4, "Sometimes": 3, "Rarely": 2, "Not at all": 1}

# Map string values to numerical values
for col in df.columns:
    if col.startswith("i12_health_"):
        df[col] = df[col].map(frequency_dict)

# Calculate the behavior scale
df["behavior_scale"] = df[["i12_health_1", "i12_health_2", "i12_health_3", "i12_health_4", "i12_health_5", "i12_health_6", "i12_health_7", "i12_health_8", "i12_health_9", "i12_health_10", "i12_health_11", "i12_health_12", "i12_health_13", "i12_health_14", "i12_health_15", "i12_health_16", "i12_health_17", "i12_health_18", "i12_health_19", "i12_health_20", "i12_health_21", "i12_health_22", "i12_health_23", "i12_health_24", "i12_health_25", "i12_health_26", "i12_health_27", "i12_health_28","i12_health_29"]].mean(axis=1)

# Visualization using matplotlib
fig, ax = plt.subplots()

# Create a box plot for the behavior scale
boxplot = df.boxplot(column='behavior_scale', by="endtime",ax=ax)

# Customize the plot
ax.set_title('Behavioral Dynamics during COVID-19 Pandemic')
ax.set_xlabel("Time")
ax.set_ylabel('Behavior Scale')

# Show the plot
plt.show()
