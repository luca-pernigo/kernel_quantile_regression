# script to create df Temperatures| Load
from datetime import datetime
import pandas as pd

import holidays


year=2022

df_load=pd.read_csv(f"Data/CH/{year}/clean/load.csv")


df_temp=pd.read_csv(f"Data/CH/{year}/clean/temperature.csv")
df_wind=pd.read_csv(f"Data/CH/{year}/clean/wind_speed.csv")


# clean timezone info
time_clean = [i[:-6] for i in df_load["Time"]]
df_load["Time"] =time_clean
df_load["Time"] = pd.to_datetime(time_clean)
df_temp["Time"] = pd.to_datetime(df_temp["Time"] )
df_wind["Time"] = pd.to_datetime(df_wind["Time"])

# merge on time
df=pd.merge(df_load, df_temp, how="right", on=["Time"])
df=pd.merge(df, df_wind, how="left", on=["Time"])


# reset index
df.reset_index(inplace=True, drop=True)


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
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)


df.to_csv(f"Data/CH/{year}/clean/ch.csv", index=False)




