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

qweek = df["qweek"]

# Create a box plot
# plt.figure(figsize=(10, 6))
# df.boxplot(column="i12_health_1", by="qweek", showfliers=False)
# plt.title("Frequency vs Time in Australia")
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.show()

# Create a line plot connecting the averages
average_mask_by_week = df.groupby(["qweek","i12_health_1"]).mean()
print(average_mask_by_week)
# plt.figure(figsize=(10, 6))
# plt.plot(average_mask_by_week.index, average_mask_by_week.values, marker='o', linestyle='-', color='red')
# plt.title("Average Frequency vs Time in Australia")
# plt.xlabel("Time")
# plt.ylabel("Average Frequency")
# plt.show()

