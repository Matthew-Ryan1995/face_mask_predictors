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
# print(df.loc[mask, "PHQ4_4"].unique())
print(df.loc[mask, "PHQ4_4"].isnull().sum())

df.loc[mask, "PHQ4_4"] = df.loc[mask, "PHQ4_4"].fillna("N/A")
print(df.loc[mask, "PHQ4_4"].isnull().sum())

# # create a new column in the csv that computer from week 1 for every two weeks

# # Find the start date (minimum date) and end date (maximum date)
# start_date = df['endtime'].min()
# end_date = df['endtime'].max()

# # Create a new column 'week_number' and assign week numbers
# df['week_number'] = ((df['endtime'] - start_date).dt.days // 14) + 1

# # Add in cleaning of other variables as well

# # Do this last
# df.dropna(inplace=True)


# # Save the cleaned DataFrame to a new CSV file
# # edit: fixed file path/save location
# df.to_csv("data/cleaned_data.csv", index=False)