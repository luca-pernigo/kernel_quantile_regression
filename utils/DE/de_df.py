# script to create df Temperatures| Load
from datetime import datetime
import pandas as pd

import holidays

df_load=pd.read_csv("Data/DE/2021/clean/load.csv")


df_temp=pd.read_csv("Data/DE/2021/clean/temperature.csv")


# merge on time
df=pd.merge(df_load, df_temp, how="right", on=["Time"])

# drop last because is NaN
df=df.iloc[:-1, :].copy()
df.reset_index(inplace=True, drop=True)

# clean timezone info
time_clean = [i[:-6] for i in df["Time"]]

df["Time"] = pd.to_datetime(time_clean)

# get additional info
df['Day'] = df["Time"].dt.day
df['Month'] = df["Time"].dt.month
df['Hour'] = df["Time"].dt.hour
    
# add info weekday; saturday, sunday
df['Day_of_week'] = df["Time"].dt.dayofweek

# # holidays
hol=holidays.DE(years=range(df["Time"].iloc[0].year, df["Time"].iloc[-1].year+1))
df["Is_holiday"]=pd.Series(df["Time"].dt.date).isin(hol).astype(int)  

# save
df.to_csv("Data/DE/2021/clean/de.csv", index=False)




