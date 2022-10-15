import datetime as dt
import pandas as pd


df = pd.read_csv('CENTRALIZED_data_record.csv')

df['Time'] = pd.to_datetime(df['Time'])
print(df['Time'].apply(lambda t: t.hour * 3600 + t.minute * 60))