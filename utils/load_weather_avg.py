import numpy as np
import pandas as pd

import holidays

import sys

import miscellaneous

# python utils/load_weather_avg.py "Data/Load/Task 1/L1-train.csv"
def weather_train(file):
    df=miscellaneous.clean_time(file, 2001, 2011,1)

    # average temperatures
    # select weather stations data using regex
    weather_stat=df.filter(regex=("w.*")).columns.to_list()
    # average them 
    df['w_avg'] = df[weather_stat].mean(axis=1)

    # drop columns
    df.drop(columns=weather_stat, inplace=True)
    
    # # order columns
    df_train=miscellaneous.order_columns(df,["LOAD","DAY","MONTH","HOUR","DAY_OF_WEEK","IS_HOLIDAY","w_avg"])
    
    # save
    df_train.to_csv(f"Data/Load/L-weather-train.csv", index=False)

if __name__=="__main__":
    file=sys.argv[1]
    # clean passed file
    weather_train(file)