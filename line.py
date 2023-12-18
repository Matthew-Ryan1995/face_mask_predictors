import pandas as pd
import matplotlib.pyplot as plt
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
    

df = pd.read_csv("raw_data/united-kingdom.csv", encoding='ISO-8859-1')  

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
print(df["endtime"])

# # Calculate the median for the behavior scale for each unique date
# df["behavior_scale"] = df[["i12_health_1", "i12_health_2", "i12_health_3", "i12_health_4", "i12_health_5", "i12_health_6", "i12_health_7", "i12_health_8", "i12_health_9", "i12_health_10", "i12_health_11", "i12_health_12", "i12_health_13", "i12_health_14", "i12_health_15", "i12_health_16", "i12_health_17", "i12_health_18", "i12_health_19", "i12_health_20", "i12_health_21", "i12_health_22", "i12_health_23", "i12_health_24", "i12_health_25", "i12_health_26", "i12_health_27", "i12_health_28","i12_health_29"]].mean(axis=1)
# df_grouped = df.groupby(["endtime"])["behavior_scale"].median()
# # df_grouped = pd.to_datetime(df_grouped.index, format = "%d/%m/%Y", errors='coerce')
# print(df_grouped)

# # Set specific dates for the x-axis ticks
# specific_dates = ["15/06/2020", "25/01/2021", "19/07/2021", "10/01/2022", "20/06/2022"]

# # Visualization using matplotlib
# fig, ax = plt.subplots()

# # Create a line plot for the median behavior scale with time on the x-axis
# plt.plot(df_grouped.index, df_grouped.values, marker='o', linestyle='-')

# # Customize the plot
# ax.set_title('Median Behavioral Dynamics over Time')
# ax.set_xlabel('Time')
# ax.set_ylabel('Median Behavior Scale')
# ax.set_xticks([convert_date_format(date) for date in specific_dates])

# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45)

# # Show the plot
# plt.tight_layout()
# plt.show()