import pandas as pd

cleaned_df = pd.read_csv("data/cleaned_data.csv", keep_default_na = False)
    
# add in one column indicates whether it is within face mask mandates
states_date = {'Australian Capital Territory': ['2021-06-28', '2022-02-25'], 'New South Wales': ['2021-01-04', '2022-09-20'], 'Northern Territory': ['2021-12-19', '2022-03-05'], 'Queensland': ['2021-06-29', '2022-03-07'], 'South Australia': ['2021-07-27', '2022-09-20'], 'Tasmania': ['2021-12-21', '2022-03-05'], 'Victoria': ['2020-07-23', '2022-09-22'], 'Western Australia': ['2021-12-23', '2022-04-29']}
    
for state, date_range in states_date.items():
    states_date[state] = [pd.to_datetime(date, format='%Y-%m-%d') for date in date_range]

# Create a new column "period"
cleaned_df['within_mandate_period'] = cleaned_df.index.map(lambda idx: idx if idx in cleaned_df.index else 0)

# Create dummy variables for 'state'
state_dummies = pd.get_dummies(cleaned_df['state'], prefix='state', drop_first=True)

# Create dummy variables for 'gender'
gender_dummies = pd.get_dummies(cleaned_df['gender'], prefix='gender', drop_first=True)

# Create dummy variables for 'employment_status'
employment_status_dummies = pd.get_dummies(cleaned_df['employment_status'], prefix='employment_status', drop_first=True)

# Concatenate the dummy variables with the original DataFrame
cleaned_df = pd.concat([cleaned_df, state_dummies, gender_dummies, employment_status_dummies], axis=1)

# Drop the original categorical columns
cleaned_df = cleaned_df.drop(['state', 'gender', 'employment_status'], axis=1)

# Store the updated DataFrame
cleaned_df.to_csv("data/cleaned_data_preprocessing.csv", index=False)