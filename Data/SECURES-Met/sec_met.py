
from datetime import datetime
import pandas as pd

import glob
import os

import sys
sys.path.append("/Users/luca/Desktop/kernel_quantile_regression")
from utils.miscellaneous import en_clean_load


# procedures used for cleaning SECURES-Met data

def clean_sec_met_train(filepath, country, field, savename):
    # script to create data
    df=pd.read_csv(f"{filepath}")
    df.rename(columns={df.columns[0]:"Unnamed: 0"}, inplace=True)
    # name col time
    # Time,Load,Temperature,Wind_speed,Day,Month,Hour,Day_of_week,Is_holiday
    df.rename(columns={"Unnamed: 0":"Time"}, inplace=True)

    # datetime
    if field=="Hydro_reservoir" or field=="Hydro_river":
        # repeat 24 times the daily data
        rep = df[country][-366:].repeat(24)
        # new df
        df= pd.DataFrame({country: rep})
        # time info
        date_range = pd.date_range(start='2020-01-01 00:00', end='2020-12-31 23:00', freq='H')
        df["Time"]=date_range

    else:
        df["Time"]=pd.to_datetime(df["Time"], format="%Y-%m-%d-%H")

    # 2020
    df=df[(df["Time"].dt.year>2019) & (df["Time"].dt.year<2021)]
    df.reset_index(inplace=True)


    # CH
    df=df[["Time", country]]
    df.rename(columns={country:field}, inplace=True)

    df.to_csv(f"Data/SECURES-Met/{country}/clean/{savename}.csv", index=False)




def clean_sec_met_test(filepath, country, field, savename):
    # script to create data
    df=pd.read_csv(f"{filepath}")
    df.rename(columns={df.columns[0]:"Unnamed: 0"}, inplace=True)
    # name col time
    # Time,Load,Temperature,Wind_speed,Day,Month,Hour,Day_of_week,Is_holiday
    df.rename(columns={"Unnamed: 0":"Time"}, inplace=True)

    # datetime
    if field=="Hydro_reservoir" or field=="Hydro_river":
        # repeat 24 times the daily data
        rep = df[country][-365:].repeat(24)
        # new df
        df= pd.DataFrame({country: rep})
        # time info
        date_range = pd.date_range(start='2021-01-01 00:00', end='2021-12-31 23:00', freq='H')
        df["Time"]=date_range

    else:
        df["Time"]=pd.to_datetime(df["Time"], format="%Y-%m-%d-%H")

    # 2020
    df=df[(df["Time"].dt.year>2020) & (df["Time"].dt.year<2022)]
    df.reset_index(inplace=True)


    # CH
    df=df[["Time", country]]
    df.rename(columns={country:field}, inplace=True)

    df.to_csv(f"Data/SECURES-Met/{country}/clean/test/{savename}.csv", index=False)





def clean_load(file, country, train_test):

    df=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/SECURES-Met/{country}/{file}.csv")

    df=en_clean_load(df)
    df.reset_index(inplace=True, drop=True)

    df.to_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/SECURES-Met/{country}/clean/{train_test}/load.csv", index=False)


def df_create(dir):
    df1=pd.read_csv(f"{dir}/direct_irradiation.csv", parse_dates=["Time"])
    df2=pd.read_csv(f"{dir}/global_radiation.csv",parse_dates=["Time"])
    df3=pd.read_csv(f"{dir}/hydro_reservoir.csv",parse_dates=["Time"])
    df4=pd.read_csv(f"{dir}/hydro_river.csv",parse_dates=["Time"])
    df5=pd.read_csv(f"{dir}/temperature.csv",parse_dates=["Time"])
    df6=pd.read_csv(f"{dir}/wind_potential.csv",parse_dates=["Time"])
    df7=pd.read_csv(f"{dir}/load.csv",parse_dates=["Time"])  

    # merge on time
    df7['Time']=pd.to_datetime(df7["Time"], utc=True)
    df7['Time']+= pd.Timedelta(hours=1)
    df7['Time']=df7['Time'].dt.tz_convert(None)

    df=pd.merge(df7, df1, how="inner", on=["Time"])
    
    res=pd.concat([df["Time"],df["Load"], df1['Direct_irradiation'], df2['Global_radiation'], df3["Hydro_reservoir"], df4["Hydro_river"], df5["Temperature"], df6["Wind_potential"]], axis=1)
    
    
    res.to_csv(f"{dir}/df.csv", index=False)