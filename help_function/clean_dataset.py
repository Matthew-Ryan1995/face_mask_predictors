# filter out those with missing values larger than 5000
import pandas as pd
import datetime as datetime
from missing_table import missing_value_df

df = pd.read_csv("raw_data/australia.csv")
# Set the threshold for missing value count
thresh_value = 1000

# Extract the variable names with missing value counts larger than 1000
columns_to_drop = missing_value_df.loc[missing_value_df['Missing Value Count'] > thresh_value, 'Variable Name'].tolist()
columns_to_drop.append('q_other') # __NA__ covers almost all of the columns
columns_to_drop.append('i14_health_other') # __NA__ covers almost all of the columns


df.drop(columns=columns_to_drop, inplace = True)
df.dropna(inplace= True)
# print(df)

# Save the cleaned DataFrame to a new CSV file
df.to_csv("cleaned_data.csv", index=False)