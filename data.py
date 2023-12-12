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

# Double Check NULL value in these variables
# for var in full_rec_vars:
#     print(f"{var : >16} has total amount of missing value of {df[var].isnull().sum()}")

# **
# Scatter plot to see how the data distribute
# # **#
plt.hist(list(mask.tolist()))
plt.xticks([1, 2, 3, 4, 5], labels=[f"{label}: {value}" for label, value in frequency_dict.items()])
plt.title("Frequency of wearing face mask in public for individuals in Australia")
plt.xlabel("Frequency")
plt.ylabel("Number of certain level of frequency")
plt.show()
# done

# Create a line plot for each frequency level
# Convert "qweek" to a categorical data type with original order
df["qweek"] = pd.Categorical(df["qweek"], categories=df["qweek"].unique(), ordered=True)

for label, freq in frequency_dict.items():
    subset = df[df["i12_health_1"] == label]
    counts_per_week = subset.groupby("qweek").size()
    plt.plot(counts_per_week.index, counts_per_week.values, marker='o', linestyle='-', label=f"{label}")

plt.title("Total Number of People for Each Frequency Level vs Time in Australia")
plt.xlabel("Time")
plt.ylabel("Total Number of People")
plt.xticks(rotation=45)
plt.legend()
plt.show()