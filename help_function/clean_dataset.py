# filter out those with missing values larger than 5000
import pandas as pd
import datetime as datetime
from missing_table import missing_value_df

##note: More needs to be added to this script to clean the columns as needs be

df = pd.read_csv("raw_data/australia.csv")
# Set the threshold for missing value count
##edit: change cound to 1006
thresh_value = 1006

# Extract the variable names with missing value counts larger than 1000
columns_to_drop = missing_value_df.loc[missing_value_df['Missing Value Count']
                                       > thresh_value, 'Variable Name'].tolist()
columns_to_drop.append('q_other')  # __NA__ covers almost all of the columns
# __NA__ covers almost all of the columns
columns_to_drop.append('i14_health_other')


df.drop(columns=columns_to_drop, inplace=True)
df.dropna(inplace=True)
# print(df)

# Save the cleaned DataFrame to a new CSV file
# edit: fixed file path/save location
df.to_csv("../data/cleaned_data.csv", index=False)
