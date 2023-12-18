import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv("raw_data/united-kingdom.csv", encoding='ISO-8859-1')  

date_time_str = df["endtime"][0]
print(date_time_str)

# Parse the string to a datetime object
date_time_obj = datetime.strptime(date_time_str, '%m/%d/%Y %H:%M')

# Extract and print the date part
formatted_date = date_time_obj.strftime('%m/%d/%Y')

print(formatted_date)