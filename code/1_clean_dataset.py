'''
It is useful to have a quick description of the scripts purpose up here.

For example:
    
    This script will read in the dataset, select relevant columns, and clean
    the data.
    
Author:
    Jinjing Ye, Matt Ryan
    
Date created:
    17/04/2024
'''

# filter out those with missing values larger than 5000
import pandas as pd
# from missing_table import missing_value_df # Change: directly import the output from the missing_table
from datetime import datetime

# This function converts the string into the type of datetime %d/%m/%Y


def convert_datetime(dt):
    date = dt.split()[0]
    return datetime.strptime(date, "%d/%m/%Y")


def household_convert(size_str):
    for i in range(1, 8):
        if size_str == str(i):
            return i
        elif size_str == "8 or more":
            return 8
        elif size_str == "Prefer not to say" or size_str == "Don't know":
            return None


# note: More needs to be added to this script to clean the columns as needs be
# edit: changed filed path
df = pd.read_csv("../raw_data/australia.csv",
                 na_values=[" ", "__NA__"], keep_default_na=True)

df["endtime"] = df["endtime"].apply(convert_datetime)
thresh_value = 10781

# Extract the variable names with missing value counts larger than 10781
missing_value_df = pd.read_csv('../data/missing_value_counts.csv')
columns_to_drop = missing_value_df.loc[missing_value_df['Missing Value Count']
                                       > thresh_value, 'Variable Name'].tolist()

df.drop(columns=columns_to_drop, inplace=True)


# Identified dates where consent for medical questions was not given
sdate = "2021-02-10"  # Change: sdate -> 10-02-2021
edate = "2021-10-18"
mask = (df["endtime"] <= edate) & (df["endtime"] >= sdate)

for i in range(1, 5):
    df.loc[mask, f"PHQ4_{i}"] = df.loc[mask, f"PHQ4_{i}"].fillna("N/A")
for i in range(1, 14):
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask,
                                            f"d1_health_{i}"].fillna("N/A")
for i in range(98, 100):
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask,
                                            f"d1_health_{i}"].fillna("N/A")

# Removing missing values
df.dropna(inplace=True)

# Convert the string into values in scale
for i in range(1, 3):
    df[f"r1_{i}"] = df[f"r1_{i}"].replace(
        {"7 - Agree": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2, "1 – Disagree": 1})  # Change: change the string to int

frequency_dict = {"Always": 5, "Frequently": 4,
                  "Sometimes": 3, "Rarely": 2, "Not at all": 1}
for column in df.columns:
    if column.startswith("i12_health_"):
        df[column] = df[column].map(frequency_dict)


# Create face mask and protective behaviour scales

# Face mask
# Change: have a face mask variable to be target in model i12_health_[1,22,23,25]
df["face_mask_behaviour_scale"] = df[["i12_health_1",
                                      "i12_health_22", "i12_health_23", "i12_health_25"]].median(axis=1)
df["face_mask_behaviour_binary"] = df["face_mask_behaviour_scale"].apply(
    lambda x: "Yes" if x >= 4 else "No")

# Protective behaviour
protective_behaviour_cols = [col for col in df if col.startswith("i12_")]

df["protective_behaviour_scale"] = df[protective_behaviour_cols].median(axis=1)
df["protective_behaviour_binary"] = df["protective_behaviour_scale"].apply(
    lambda x: "Yes" if x >= 4 else "No")

# Protective behaviour, no face mask
protective_behaviour_nomask_cols = [col for col in protective_behaviour_cols if not col in ["i12_health_1",
                                                                                            "i12_health_22", "i12_health_23", "i12_health_25"]]
df["protective_behaviour_nomask_scale"] = df[protective_behaviour_nomask_cols].median(
    axis=1)

# Comorbidities: D1 Health
# Combine comorbidities into "Yes", "No, "NA", "Prefer not to say"
d1_cols = [col for col in df if col.startswith("d1_")]

df["d1_comorbidities"] = "Yes"
df.loc[df["d1_health_99"] == "Yes", "d1_comorbidities"] = "No"
df.loc[df["d1_health_99"] == "N/A", "d1_comorbidities"] = "NA"
df.loc[df["d1_health_98"] == "Yes", "d1_comorbidities"] = "Prefer_not_to_say"

df = df.drop(d1_cols, axis=1)


# Week variable in original data changes definition, so make our own.
# create a new column in the csv that computer from week 1 for every two weeks
start_date = df['endtime'].min()
end_date = df['endtime'].max()

# Create a new column 'week_number' and assign week numbers
df['week_number'] = ((df['endtime'] - start_date).dt.days // 14) + 1


# Update: household to numeric and n/a to see whether loss data/ might categorical? 0-2, 2-4
# Remove induced missing values
df["household_size"] = df["household_size"].apply(household_convert)
df.dropna(inplace=True)

# Remove weight (survey weighting) and qweek
# Also remove protective behaviour scales, not to be used in modelling
df = df.drop(["qweek", "weight",] + protective_behaviour_cols, axis=1)


# Save the cleaned DataFrame to a new CSV file
# edit: fixed file path/save location
df.to_csv("../data/cleaned_data.csv", index=False)
