import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import seaborn as sns

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
    
# N/A and mean of the column related to the mask
df = pd.read_csv("raw_data/australia_a.csv")

frequency_dict = {"Always":5, "Frequently":4, "Sometimes":3, "Rarely":2, "Not at all":1}
df["i12_health_1"] = df["i12_health_1"].map(frequency_dict)

# Convert "endtime" to datetime format
df["endtime"] = df["endtime"].apply(convert_date_format)
df["endtime"] = pd.to_datetime(df["endtime"])
start_date = df['endtime'].min()
end_date = df['endtime'].max()
print(end_date)

# Create a new column 'week_number' and assign week numbers
df['week_number'] = ((df['endtime'] - start_date).dt.days // 14) + 1
week_number = df["week_number"]
# States and their start/end times
states_data = {
    "Australian Capital Territory": ["28/6/2021", "25/2/2022"],
    "New South Wales": ["4/1/2021", "20/9/2022"],
    "Northern Territory": ["19/12/2021", "5/3/2022"],  # Add the actual values
    "Queensland": ["18/12/2021", "7/3/2022"],
    "South Australia": ["27/7/2021", "20/9/2022"],
    "Tasmania": ["21/12/2021", "5/3/2022"],
    "Victoria": ["16/8/2020", "22/9/2022"],
    "Western Australia": ["23/4/2021", "9/9/2022"]
}

# ** 
# Create a line plot for each frequency level
# ** #

#states
# Get unique states in the dataset
unique_states = df["state"].unique()

# Calculate the number of rows and columns for subplots
num_rows = len(unique_states) // 2 + len(unique_states) % 2
num_cols = 2

# Create subplots for each state
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True)

for i, state in enumerate(unique_states):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    for label, freq in frequency_dict.items():
        df_state = df[df["state"] == state]
        mean_per_week = df_state.groupby("week_number")["i12_health_1"].mean()
        ax.plot(mean_per_week.index, mean_per_week.values, marker='o', linestyle='-')
        # Box plot
        # sns.boxplot(x=df_state["week_number"], y=df_state["i12_health_1"], ax=ax, color='gray', width=0.2, showfliers=False)
    # Add vertical lines for start and end times
    start_time, end_time = states_data[state]
    if start_time:
        start_time = pd.to_datetime(start_time, format='%d/%m/%Y')
        ax.axvline(x=((start_time - start_date).days // 14) + 1, color='green', linestyle='--', label='Face mask mandate Starts')
    if end_time:
        end_time = pd.to_datetime(end_time, format='%d/%m/%Y')
        ax.axvline(x=((end_time - start_date).days // 14) + 1, color='red', linestyle='--', label='Face mask mandate Ends')

    ax.set_title(f"Behavior Dynamic in wearing the mask outside home in {state}")
    ax.set_ylabel("Mean of the behavior scale\nin wearing the mask outside home")
    ax.set_ylim(0,5)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.set_xlabel("Time in every two weeks (from 01/04/2021 to 28/03/2022)")

# Set common X-axis label
# plt.xlabel("Time (in weeks)")

# Adjust layout
plt.tight_layout()
plt.show()
# done