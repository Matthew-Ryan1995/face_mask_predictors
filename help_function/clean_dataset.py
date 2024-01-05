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
from missing_table import missing_value_df
from datetime import datetime

# This function converts the string into the type of datetime %d/%m/%Y


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

# note: More needs to be added to this script to clean the columns as needs be


# edit: changed filed path
df = pd.read_csv("raw_data/australia.csv",
                 na_values=[" ", "__NA__"], keep_default_na=True)
# Set the threshold for missing value count
# edit: change cound to 10781
thresh_value = 10781

# Extract the variable names with missing value counts larger than 1000
columns_to_drop = missing_value_df.loc[missing_value_df['Missing Value Count']
                                       > thresh_value, 'Variable Name'].tolist()

df.drop(columns=columns_to_drop, inplace=True)

# Add N/A category to d1 and PHQ columns for specific weeks (from 19/02/21 week25 to 18/10/21 week43)
# Add n/a category to d1 and PHQ to clean the dataset to add in another category in weeks 
df["endtime"] = df["endtime"].apply(convert_date_format)
df["endtime"] = pd.to_datetime(df["endtime"])
sdate = "19-02-2021"
edate = "18-10-2021"
mask = (df["endtime"] <= edate) & (df["endtime"] >= sdate)

# Clean the d1_health and PHQ4
phq_answer = {"Not at all": 1, "Several days": 2, "More than half the days": 3, "Nearly every day": 4, "Prefer not to say": 5}
for i in range(1,5):
    # print(df.loc[mask, f"PHQ4_{i}"].isnull().sum())
    df.loc[mask, f"PHQ4_{i}"] = df.loc[mask, f"PHQ4_{i}"].fillna("N/A")
    df[f"PHQ4_{i}"] = df[f"PHQ4_{i}"].map(phq_answer)
    # print(df.loc[mask, f"PHQ4_{i}"].isnull().sum())

d1_answer = {"Yes": 1, "No": 0, "N/A": 99}
for i in range(1,14):
    # print(df.loc[mask, f"d1_health_{i}"].isnull().sum())
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask, f"d1_health_{i}"].fillna("N/A")
    df[f"d1_health_{i}"] = df[f"d1_health_{i}"].map(d1_answer)
    # print(df.loc[mask, f"d1_health_{i}"].isnull().sum())

for i in range(98,100):
    # print(df.loc[mask, f"d1_health_{i}"].isnull().sum())
    df.loc[mask, f"d1_health_{i}"] = df.loc[mask, f"d1_health_{i}"].fillna("N/A")
    df[f"d1_health_{i}"] = df[f"d1_health_{i}"].map(d1_answer)
    # print(df.loc[mask, f"d1_health_{i}"].isnull().sum())

wcr_answer = {"Very well":1, "Somewhat well": 2, "Somewhat badly": 3, "Very badly": 4, "Don't know": 5}
for i in range(1,3):
    df[f"WCRex{i}"] = df[f"WCRex{i}"].map(wcr_answer)
# Add in cleaning of other variables as well    
# Drop the useless columns
df = df.drop(["RecordNo", "household_size", "qweek", "weight"], axis = 1)

# Convert the string into values in scale
for i in range(1, 3):
    df[f"r1_{i}"] = df[f"r1_{i}"].replace({"7 - Agree": 7, "1 – Disagree": 1})

frequency_dict = {"Always": 5, "Frequently": 4, "Sometimes": 3, "Rarely": 2, "Not at all": 1}
for column in df.columns:
    if column.startswith("i12_health_"):
        df[column] = df[column].map(frequency_dict)

i9_answer = {"Yes": 1, "No": 0, "Not sure": 99}
df["i9_health"] = df["i9_health"].map(i9_answer)

i11_answer = {"Very willing": 1, "Somewhat willing": 2, "Neither willing nor unwilling": 3 ,"Somewhat unwilling": 4, "Very unwilling": 5, "Not sure": 6}
df["i11_health"] = df["i11_health"].map(i11_answer)

employ_answer = {"Full time employment": 1, "Part time employment": 2, "Full time student": 3, "Retired": 4, "Unemployed": 5, "Not working":6, "Other":7}
df["employment_status"] = df["employment_status"].map(employ_answer)
# Compare those columns with similar factor i.e. face mask

    
# create a new column in the csv that computer from week 1 for every two weeks

# Find the start date (minimum date) and end date (maximum date)
start_date = df['endtime'].min()
end_date = df['endtime'].max()

# Create a new column 'week_number' and assign week numbers
df['week_number'] = ((df['endtime'] - start_date).dt.days // 14) + 1



# Do this last
df.dropna(inplace=True)

print(df["r1_1"].value_counts().index.tolist())
# Save the cleaned DataFrame to a new CSV file
# edit: fixed file path/save location
df.to_csv("data/cleaned_data.csv", index=False)