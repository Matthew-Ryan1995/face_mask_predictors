import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime

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
df = pd.read_csv("raw_data/australia.csv")

frequency_dict = {"Always": 5, "Frequently": 4, "Sometimes": 3, "Rarely": 2, "Not at all": 1}
mask = df["i12_health_1"]

mask = mask.map(frequency_dict)

# Convert "endtime" to datetime format
df["endtime"] = df["endtime"].apply(convert_date_format)
df["endtime"] = pd.to_datetime(df["endtime"])
start_date = df['endtime'].min()
end_date = df['endtime'].max()

# Create a new column 'week_number' and assign week numbers
df['week_number'] = ((df['endtime'] - start_date).dt.days // 14) + 1
week_number = df["week_number"]
# States and their start/end times 
states_data = {
    "Australian Capital Territory": ["28/6/2021", "25/2/2022"],
    "New South Wales": ["4/1/2021", "20/9/2022"],
    "Northern Territory": ["19/12/2021", "5/3/2022"],
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
        subset = df_state[df_state["i12_health_1"] == label]
        count_per_week = subset.groupby("week_number", observed=False).size()
        ax.plot(count_per_week.index, count_per_week.values, marker='o', linestyle='-', label=f"{label}")

    # Add vertical lines for start and end times
    start_time, end_time = states_data[state]
    if start_time:
        start_time = pd.to_datetime(start_time, format='%d/%m/%Y')
        ax.axvline(x=((start_time - start_date).days // 14) + 1, color='green', linestyle='--', label='Face mask mandate Starts')
    if end_time and (state == "Australian Capital Territory" or state == "Northen Territory" or state == "Queensland" or state == "Tasmania"):
        end_time = pd.to_datetime(end_time, format='%d/%m/%Y')
        ax.axvline(x=((end_time - start_date).days // 14) + 1, color='red', linestyle='--', label='Face mask mandate Ends')

    ax.set_title(f"Total Number of People for Each Frequency Level vs Time in {state}")
    ax.set_ylabel("Total Number of People \nin wearing the mask outside home")

    ax.tick_params(axis='x', rotation=45)
    ax.legend()

# Set common X-axis label
plt.xlabel("Time in every two weeks (from 01/04/2021 to 28/03/2022)")

# Adjust layout
plt.tight_layout()
plt.show()
# done