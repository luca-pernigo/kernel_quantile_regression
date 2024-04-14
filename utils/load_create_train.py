import numpy as np
import pandas as pd

import holidays

import sys

import miscellaneous

# python utils/load_create_train.py "Data/Load/Task 1/L1-train.csv"
def create_train(file):
    df=miscellaneous.clean_time(file, 2001, 2011,1)

    # subset from which we have load data
    df=df[df["YEAR"]>=2005][1:]

    # create DS column
    df["DS"]=pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"], hour=df["HOUR"]))

    # average temperatures
    # select weather stations data using regex
    weather_stat=df.filter(regex=("w.*")).columns.to_list()
    # average them 
    df['w_avg'] = df[weather_stat].mean(axis=1)

    # drop columns
    df.drop(columns=weather_stat, inplace=True)
    
    # # order columns
    df_train=miscellaneous.order_columns(df,["LOAD","DAY","MONTH","HOUR","YEAR","DAY_OF_WEEK","IS_HOLIDAY","w_avg","DS"])
    
    # save
    df_train.to_csv(f"Data/Load/L-train.csv", index=False)

if __name__=="__main__":
    file=sys.argv[1]
    # clean passed file
    create_train(file)