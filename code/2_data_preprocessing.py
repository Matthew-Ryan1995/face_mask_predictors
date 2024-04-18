'''
Preprocess data for modelling
    
Author:
    Jinjing Ye, Matt Ryan
    
Date created:
    17/04/2024
'''

import pandas as pd


def mandates_convert(row):
    endtime = pd.to_datetime(row['endtime'], format='%Y-%m-%d')
    state = row['state']

    if states_date[state][0] <= endtime <= states_date[state][1]:
        return 1
    else:
        return 0


cleaned_df = pd.read_csv("../data/cleaned_data.csv", keep_default_na=False)
# print(cleaned_df['d1_health_1'].value_counts())

# add in one column indicates whether it is within face mask mandates
# Dates taken from media/news outlets
states_date = {'Australian Capital Territory': ['2021-06-28', '2022-02-25'],
               'New South Wales': ['2021-01-04', '2022-09-20'],
               'Northern Territory': ['2021-12-19', '2022-03-05'],
               'Queensland': ['2021-06-29', '2022-03-07'],
               'South Australia': ['2021-07-27', '2022-09-20'],
               'Tasmania': ['2021-12-21', '2022-03-05'],
               'Victoria': ['2020-07-23', '2022-09-22'],
               'Western Australia': ['2021-12-23', '2022-04-29']}

for state, date_range in states_date.items():
    states_date[state] = [pd.to_datetime(
        date, format='%Y-%m-%d') for date in date_range]

# Create a new column "period"
cleaned_df['within_mandate_period'] = cleaned_df.apply(
    mandates_convert, axis=1)

# Create dummy variables with these with more than 2 answers
convert_into_dummy_cols = ['state', 'gender', 'i9_health', 'employment_status', 'i11_health',
                           'WCRex1', 'WCRex2', 'PHQ4_1', 'PHQ4_2', 'PHQ4_3', 'PHQ4_4',
                           'd1_health_1', 'd1_health_2', 'd1_health_3', 'd1_health_4', 'd1_health_5',
                           'd1_health_6', 'd1_health_7', 'd1_health_8', 'd1_health_9', 'd1_health_10',
                           'd1_health_11', 'd1_health_12', 'd1_health_13', 'd1_health_98', 'd1_health_99']

for col in convert_into_dummy_cols:
    dummy = pd.get_dummies(cleaned_df[col], prefix=col, drop_first=True)
    cleaned_df = pd.concat([cleaned_df, dummy], axis=1)
    cleaned_df = cleaned_df.drop(col, axis=1)

# todo: Scaling?

# Store the updated DataFrame
cleaned_df.to_csv("../data/cleaned_data_preprocessing.csv", index=False)
