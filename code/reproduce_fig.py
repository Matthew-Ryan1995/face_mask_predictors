import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

# Function to convert the date format
def convert_date_format(date_str):
    try:
        date_time_obj = datetime.strptime(date_str, '%d/%m/%Y %H:%M')
        formatted_date = date_time_obj.strftime('%m/%d/%Y')
        return formatted_date
    except ValueError:
        pass

    try:
        date_time_obj = datetime.strptime(date_str, '%d/%m/%Y')
        return date_str
    except ValueError:
        raise ValueError(f"Unsupported date format: {date_str}")

df = pd.read_csv("raw_data/australia.csv", encoding='ISO-8859-1')  

# Frequency dictionary
frequency_dict = {"Always": 5, "Frequently": 4, "Sometimes": 3, "Rarely": 2, "Not at all": 1}

# Map string values to numerical values
for col in df.columns:
    if col.startswith("i12_health_"):
        df[col] = df[col].map(frequency_dict)

# convert "endtime" with date %m/%d/%Y
df["endtime"] = df["endtime"].apply(convert_date_format)
df["endtime"] = pd.to_datetime(df["endtime"])   
df = df.sort_values(by=["endtime"])
# Calculate the median for the behavior scale for each unique date
df["behavior_scale"] = df[["i12_health_1", "i12_health_2", "i12_health_3", "i12_health_4", "i12_health_5", "i12_health_6", "i12_health_7", "i12_health_8", "i12_health_9", "i12_health_10", "i12_health_11", "i12_health_12", "i12_health_13", "i12_health_14", "i12_health_15", "i12_health_16", "i12_health_17", "i12_health_18", "i12_health_19", "i12_health_20", "i12_health_21", "i12_health_22", "i12_health_23", "i12_health_24", "i12_health_25", "i12_health_26", "i12_health_27", "i12_health_28","i12_health_29"]].mean(axis=1)

# Filter out the day with less than 500/300 data points
df_grouped_size = df.groupby("endtime").size()

# Filter out groups with sizes less than 500/300
valid_endtimes = df_grouped_size[df_grouped_size >= 300].index

# Filter the original DataFrame to keep only rows with valid endtimes
df_filtered = df[df["endtime"].isin(valid_endtimes)]
# Calculate the median for the filtered DataFrame
df_grouped = df_filtered.groupby("endtime")["behavior_scale"].median()

# Create a box plot using seaborn
df_filtered.index = df_filtered.index.astype(str)
df_grouped.index = df_grouped.index.astype(str)

sns.set()

plt.figure(figsize=(12, 6))
ax = sns.boxplot(x=df_filtered["endtime"], y=df_filtered["behavior_scale"])
ax = sns.lineplot(x=df_grouped.index, y=df_grouped.values, linestyle='-', linewidth = 1.5, color='g')
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
# plt.title('Behavioral Dynamics over Time in UK')
plt.title('Behavioral Dynamics over Time in AU')
plt.xlabel('Time')
plt.ylabel('Behavior Scale')
plt.tight_layout()
plt.show()

# change this plot to proportion of male and female