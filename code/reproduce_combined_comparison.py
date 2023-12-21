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

# Read the UK data
df_uk = pd.read_csv("raw_data/united-kingdom.csv", encoding='ISO-8859-1')  

# Frequency dictionary
frequency_dict = {"Always": 5, "Frequently": 4, "Sometimes": 3, "Rarely": 2, "Not at all": 1}

# Map string values to numerical values
for col in df_uk.columns:
    if col.startswith("i12_health_"):
        df_uk[col] = df_uk[col].map(frequency_dict)

# Convert "endtime" to datetime format
df_uk["endtime"] = df_uk["endtime"].apply(convert_date_format)
df_uk["endtime"] = pd.to_datetime(df_uk["endtime"])   
df_uk = df_uk.sort_values(by=["endtime"])

# Calculate the median for the behavior scale for each unique date
df_uk["behavior_scale"] = df_uk[["i12_health_1", "i12_health_2", "i12_health_3", "i12_health_4", "i12_health_5", "i12_health_6", "i12_health_7", "i12_health_8", "i12_health_9", "i12_health_10", "i12_health_11", "i12_health_12", "i12_health_13", "i12_health_14", "i12_health_15", "i12_health_16", "i12_health_17", "i12_health_18", "i12_health_19", "i12_health_20", "i12_health_21", "i12_health_22", "i12_health_23", "i12_health_24", "i12_health_25", "i12_health_26", "i12_health_27", "i12_health_28","i12_health_29"]].mean(axis=1)

# Filter out the day with less than 500 data points
df_grouped_size_uk = df_uk.groupby("endtime").size()

# Filter out groups with sizes less than 500
valid_endtimes_uk = df_grouped_size_uk[df_grouped_size_uk >= 500].index

# Filter the original DataFrame to keep only rows with valid endtimes
df_filtered_uk = df_uk[df_uk["endtime"].isin(valid_endtimes_uk)]

# Calculate the median for the filtered DataFrame
df_grouped_uk = df_filtered_uk.groupby("endtime")["behavior_scale"].median()


# Read the Australia data
df_australia = pd.read_csv("raw_data/australia.csv", encoding='ISO-8859-1')  

# Frequency dictionary
frequency_dict_au = {"Always": 5, "Frequently": 4, "Sometimes": 3, "Rarely": 2, "Not at all": 1}

# Map string values to numerical values
for col in df_australia.columns:
    if col.startswith("i12_health_"):
        df_australia[col] = df_australia[col].map(frequency_dict_au)

# Convert "endtime" to datetime format
df_australia["endtime"] = df_australia["endtime"].apply(convert_date_format)
df_australia["endtime"] = pd.to_datetime(df_australia["endtime"])   
df_australia = df_australia.sort_values(by=["endtime"])

# Calculate the median for the behavior scale for each unique date
df_australia["behavior_scale"] = df_australia[["i12_health_1", "i12_health_2", "i12_health_3", "i12_health_4", "i12_health_5", "i12_health_6", "i12_health_7", "i12_health_8", "i12_health_9", "i12_health_10", "i12_health_11", "i12_health_12", "i12_health_13", "i12_health_14", "i12_health_15", "i12_health_16", "i12_health_17", "i12_health_18", "i12_health_19", "i12_health_20", "i12_health_21", "i12_health_22", "i12_health_23", "i12_health_24", "i12_health_25", "i12_health_26", "i12_health_27", "i12_health_28","i12_health_29"]].mean(axis=1)

# Filter out the day with less than 500 data points
df_grouped_size_au = df_australia.groupby("endtime").size()

# Filter out groups with sizes less than 500
valid_endtimes_au = df_grouped_size_au[df_grouped_size_au >= 300].index

# Filter the original DataFrame to keep only rows with valid endtimes
df_filtered_au = df_australia[df_australia["endtime"].isin(valid_endtimes_au)]

# Calculate the median for the filtered DataFrame
df_grouped_au = df_filtered_au.groupby("endtime")["behavior_scale"].median()

# Create a combined plot using Seaborn and Matplotlib
plt.figure(figsize=(12, 10))
sns.set()

# Plot the line plot for the UK on the first subplot
# plt.subplot(2, 1, 1)
df_filtered_uk.index = df_filtered_uk.index.astype(str)
df_grouped_uk.index = df_grouped_uk.index.astype(str)
sns.boxplot(x=df_filtered_uk["endtime"], y=df_filtered_uk["behavior_scale"], color='tab:blue', width=0.4, showfliers=False)
sns.lineplot(x=df_grouped_uk.index, y=df_grouped_uk.values, marker='o', linestyle='-', label='UK Median Behavior Scale', color='b')
plt.title('Behavioral Dynamics over Time in UK and Australia')
# plt.ylabel('Behavior Scale (UK)')
# plt.ylim(0, 5)
# plt.legend()
# plt.xticks(rotation=45)

# Plot the line plot for Australia on the second subplot
# plt.subplot(2, 1, 2)
df_filtered_au.index = df_filtered_au.index.astype(str)
df_grouped_au.index = df_grouped_au.index.astype(str)
sns.boxplot(x=df_filtered_au["endtime"], y=df_filtered_au["behavior_scale"], color='tab:orange', width=0.4, showfliers=False)
sns.lineplot(x=df_grouped_au.index, y=df_grouped_au.values, marker='o', linestyle='-', label='Australia Median Behavior Scale', color='g')
plt.xlabel('Time')
plt.ylabel('Behavior Scale')
plt.ylim(0, 5)
plt.legend()

# Customize the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
