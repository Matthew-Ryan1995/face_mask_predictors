'''
It is useful to have a quick description of the scripts purpose up here.

For example:
    
    This script will read in the dataset, select relevant columns, and clean
    the data.
    
Author:
    Jinjing Ye
    
Date created:
    23/12/2023
'''

# filter out those with missing values larger than 5000
import pandas as pd
# from missing_table import missing_value_df # Change: directly import the output from the missing_table
from datetime import datetime

# This function converts the string into the type of datetime %d/%m/%Y


def convert_date_format(date_str):
    try:
        date_time_obj = datetime.strptime(date_str, '%d/%m/%Y %H:%M')
        formatted_date = date_time_obj.strftime('%d/%m/%Y')
        return formatted_date
    except ValueError:
        pass

    try:
        date_time_obj = datetime.strptime(date_str, '%d/%m/%Y')
        return date_str
    except ValueError:
        raise ValueError(f"Unsupported date format: {date_str}")

# note: More needs to be added to this script to clean the columns as needs be


# edit: changed filed path
df = pd.read_csv("raw_data/australia.csv",
                 na_values=[" ", "__NA__"], keep_default_na=True)
# Set the threshold for missing value count
# edit: change cound to 10781
thresh_value = 10781

# Extract the variable names with missing value counts larger than 10781
missing_value_df = pd.read_csv('data/missing_value_counts.csv')
columns_to_drop = missing_value_df.loc[missing_value_df['Missing Value Count']
                                       > thresh_value, 'Variable Name'].tolist()

df.drop(columns=columns_to_drop, inplace=True)

# Convert the string into values in scale
for i in range(1, 3):
    df[f"r1_{i}"] = df[f"r1_{i}"].replace({"7 - Agree": 7, "6": 6, "5": 5, "4":4, "3": 3, "2": 2, "1 – Disagree": 1}) # Change: change the string to int

frequency_dict = {"Always": 5, "Frequently": 4, "Sometimes": 3, "Rarely": 2, "Not at all": 1}
for column in df.columns:
    if column.startswith("i12_health_"):
        df[column] = df[column].map(frequency_dict)

# Add n/a category to d1 and PHQ to clean the dataset to add in another category in weeks (from 10/02/21 week25 to 18/10/21 week43)
df["endtime"] = df["endtime"].apply(convert_date_format)
df["endtime"] = pd.to_datetime(df["endtime"])
sdate = "10-02-2021" # Change: sdate -> 10-02-2021
edate = "18-10-2021"
mask = (df["endtime"] <= edate) & (df["endtime"] >= sdate)
# Update: add n/a category
for i in range(1,5):
    df.loc[mask, f"PHQ4_{i}"] = df.loc[mask, f"PHQ4_{i}"].fillna("N/A")

for i in range(1,14):
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask, f"d1_health_{i}"].fillna("N/A")

for i in range(98,100):
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask, f"d1_health_{i}"].fillna("N/A")

# Update: household to numeric and n/a to see whether loss data/ might categorical? 0-2, 2-4
for index, row in df.iterrows():
    for col in df.columns:
        if col == "household_size":
            if row[col] == "8 or more":
                df["household_size"] = 8
            elif row[col] == "":


# Change: squash d1_health into one column
df["d1_health_atleastone"] = "No"
for index, row in df.iterrows():
    for col in df.columns:
        if col.startswith("d1_health") and col.endswith("1" or "2" or "3"or "4"or "5"or "6"or "7"or "8"or "9"or "10"or "11"or "12"or "13"):
            if row[col] == "Yes":
                df.at[index, "d1_health_atleastone"] = "Yes"
                break  # Exit the loop once a value of 1 is found
        elif col.startswith("d1_health") and col.endswith("98"):
            if row[col] == "Yes":
                df.at[index, "d1_health_atleastone"] = "Prefer not to say"
        elif col.startswith("d1_health") and col.endswith("99"):
            if row[col] == "Yes":
                df.at[index, "d1_health_atleastone"] = "None of these"
    

# Change: have a face mask variable to be target in model i12_health_[1,22,23,25]
df["face_mask_behaviour_scale"] = df[["i12_health_1", "i12_health_22", "i12_health_23", "i12_health_25"]].median(axis = 1)

df["face_mask_behaviour_binary"] = df["face_mask_behaviour_scale"].apply(lambda x: "Yes" if x >= 4 else "No")

# Change: compare those wear mask with diagonsed and without diagonosed
            

# Add in cleaning of other variables as well    
# Drop the useless columns
df = df.drop(["RecordNo", "qweek", "weight"], axis = 1)
    
# create a new column in the csv that computer from week 1 for every two weeks

# Find the start date (minimum date) and end date (maximum date)
start_date = df['endtime'].min()
end_date = df['endtime'].max()

# Create a new column 'week_number' and assign week numbers
df['week_number'] = ((df['endtime'] - start_date).dt.days // 14) + 1

# Do this last 34452
df.dropna(inplace=True)

# Save the cleaned DataFrame to a new CSV file
# edit: fixed file path/save location
df.to_csv("data/cleaned_data.csv", index=False)