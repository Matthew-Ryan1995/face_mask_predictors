import pandas as pd
import matplotlib.pyplot as plt
import csv
first_line = True

# show headers
# csv_reader = csv.reader(open("raw_data/australia.csv"))

# for row in csv_reader:
#     if first_line:
#         first_line = False
#     else:
        # check N/A number and the mean of the question answer

# N/A and mean of the column related to the mask
df = pd.read_csv("raw_data/australia_a.csv")

frequency_dict = {"Always":1, "Frequently":2, "Sometimes":3, "Rarely":4, "Not at all":5}
mask = df["i12_health_1"]

mask = mask.map(frequency_dict)
print("i12_health_1: No. of N/A is", mask.isnull().sum(), "and the mean is", round(mask.mean(),2))
print("\n")

# **
# Time maner plot
# **#

qweek = df["qweek"]
# Create a box plot
# plt.figure(figsize=(10, 6))
# df.boxplot(column="i12_health_1", by="qweek", showfliers=False)
# plt.title("Frequency vs Time in Australia")
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.show()

# Create a line plot connecting the averages
# average_mask_by_week = df.groupby(["qweek","i12_health_1"]).mean()
# print(average_mask_by_week)
# plt.figure(figsize=(10, 6))
# plt.plot(average_mask_by_week.index, average_mask_by_week.values, marker='o', linestyle='-', color='red')
# plt.title("Average Frequency vs Time in Australia")
# plt.xlabel("Time")
# plt.ylabel("Average Frequency")
# plt.show()

# **
# Find the column with the same amount
# **#
csv_reader2 = csv.reader(open("raw_data/when.csv"))
first_line = True
for row in csv_reader2:
    if first_line: #once
        full_rec_vars = []
        first_line = False
    else:
        first_col = True
        for col in row:
            if first_col:
                count = 0
                var_name = col
                first_col = False
            else:
                if col == "TRUE":
                    count += 1
        if count == 49:
            full_rec_vars.append(var_name)
            # print(f"The variable called {var_name:>16}, its amount of TRUE is {count}.")
print(full_rec_vars)
# Double Check NULL value in these variables
for var in full_rec_vars:
    print(f"{var : >16} has total amount of missing value of {df[var].isnull().sum()}")

# **
# Scatter plot to see how the data distribute
# # **#
# plt.hist(list(mask.tolist()))
# plt.xticks([1, 2, 3, 4, 5], labels=[f"{label}: {value}" for label, value in frequency_dict.items()])
# plt.title("Frequency of wearing face mask in public for individuals in Australia")
# plt.xlabel("Frequency")
# plt.ylabel("Number of certain level of frequency")
# plt.show()
# done

# ** 
# Create a line plot for each frequency level
# ** #

# Convert "qweek" to a categorical data type with original order
df["qweek"] = pd.Categorical(df["qweek"], categories=df["qweek"].unique(), ordered=True)

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
        counts_per_week = subset.groupby("qweek", observed=False).size()
        ax.plot(counts_per_week.index, counts_per_week.values, marker='o', linestyle='-', label=f"{label}")

    ax.set_title(f"Total Number of People for Each Frequency Level vs Time in {state}")
    ax.set_ylabel("Total Number of People")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

# Set common X-axis label
plt.xlabel("Time")

# Adjust layout
plt.tight_layout()
# plt.show()
# done

# gender
# Get unique gender in the dataset
unique_gender = df["gender"].unique()

# Create subplots for each gender
fig, axes = plt.subplots(len(unique_gender), 1, figsize=(10, 5 * len(unique_gender)), sharex=True)

for i, gender in enumerate(unique_gender):
    ax = axes[i]
    for label, freq in frequency_dict.items():
        df_gender = df[df["gender"] == gender]
        subset = df_gender[df_gender["i12_health_1"] == label]
        counts_per_week = subset.groupby("qweek", observed=False).size()
        ax.plot(counts_per_week.index, counts_per_week.values, marker='o', linestyle='-', label=f"{label}")

    ax.set_title(f"Total Number of People for Each Frequency Level vs Time in {gender}")
    ax.set_ylabel("Total Number of People")
    ax.legend()

# Set common X-axis label
plt.xticks(rotation=45)
plt.xlabel("Time")

# Adjust layout
plt.tight_layout()

# age_cat
# Get unique age_cat in the dataset
# unique_age = df["age_cat"].unique()

# # Create subplots for each age
# fig, axes = plt.subplots(len(unique_age), 1, figsize=(10, 5 * len(unique_age)), sharex=True)

# for i, age in enumerate(unique_age):
#     ax = axes[i]
#     for label, freq in frequency_dict.items():
#         df_age = df[df["age_cat"] == age]
#         subset = df_age[df_age["i12_health_1"] == label]
#         counts_per_week = subset.groupby("qweek", observed=False).size()
#         ax.plot(counts_per_week.index, counts_per_week.values, marker='o', linestyle='-', label=f"{label}")

#     ax.set_title(f"Total Number of People for Each Frequency Level vs Time in {age}")
#     ax.set_ylabel("Total Number of People")
#     ax.legend()

# # Set common X-axis label
# plt.xlabel("Time")

# # Adjust layout
# plt.tight_layout()
# plt.show()