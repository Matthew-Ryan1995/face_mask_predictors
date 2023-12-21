'''
This script will read in the dataset, find the missing values for each variable, and output into a csv file named "data/missing_value_counts.csv".
    
Author:
    Jinjing Ye
    
Date created:
    23/12/2023
'''

import pandas as pd
    
# edit added ../ so code would run
df = pd.read_csv("raw_data/australia.csv",
                 na_values=[" ", "__NA__"])  # edit: This makes pandas read these values in the csv as missing

# Double Check NULL value in these variables
missing_value_counts = {}
for col in df:

    # edit: We can now just count which columns have missing values
    missing_count = (df[col].isna()).sum()

    # Store the information in the dictionary
    missing_value_counts[col] = missing_count

# Convert the dictionary to a DataFrame and sort it by missing value count
missing_value_df = pd.DataFrame(list(missing_value_counts.items()), columns=[
                                'Variable Name', 'Missing Value Count'])
missing_value_df = missing_value_df.sort_values(
    by=['Missing Value Count', 'Variable Name'])

# Save the DataFrame to a CSV file
# edit: File path changed so things save properly
missing_value_df.to_csv('data/missing_value_counts.csv', index=False)
