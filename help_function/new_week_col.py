import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta

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
    
df = pd.read_csv("raw_data/australia_a.csv")

# create a new column in the csv that computer from week 1 for every week
# Assuming df is your DataFrame with the "endtime" column
# Convert "endtime" to datetime if it's not already
df["endtime"] = df["endtime"].apply(convert_date_format)
df["endtime"] = pd.to_datetime(df["endtime"])   

# Find the start date (minimum date) and end date (maximum date)
start_date = df['endtime'].min()
end_date = df['endtime'].max()

# Create a new column 'week_number' and assign week numbers
df['week_number'] = ((df['endtime'] - start_date).dt.days // 14) + 1

# Print the DataFrame with the added 'week_number' column
print(df)