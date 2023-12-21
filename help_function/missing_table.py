import pandas as pd
import matplotlib.pyplot as plt
import csv
first_line = True
# edit added ../ so code would run
df = pd.read_csv("../raw_data/australia_a.csv",
                 na_values=[" ", "__NA__"])  # edit: This makes pandas read these values in the csv as missing

# Double Check NULL value in these variables
missing_value_counts = {}
for col in df:
    # Count missing values in each column
    # missing_count = (df[col] == " ").sum()

    # edit: We can now just count which columns have missing values
    missing_count = (df[col].isna()).sum()

    # Store the information in the dictionary
    missing_value_counts[col] = missing_count

    # print(f"{col : >16} has total amount of missing value of {df[col].isnull().sum()}")


# Convert the dictionary to a DataFrame and sort it by missing value count
missing_value_df = pd.DataFrame(list(missing_value_counts.items()), columns=[
                                'Variable Name', 'Missing Value Count'])
missing_value_df = missing_value_df.sort_values(
    by=['Missing Value Count', 'Variable Name'])

# Save the DataFrame to a CSV file
# edit: File path changed so things save properly
missing_value_df.to_csv('../data/missing_value_counts.csv', index=False)
