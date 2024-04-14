

import pandas as pd
import sys

import holidays

import miscellaneous
# bash script
# for ((i=1;i<=15;i++)); do python utils/price_create_train.py "Data/Price/Task $i/Task${i}_P.csv"; done

def create_train(file):
    df=miscellaneous.clean_time(file, 2011, 2014,0)

    df_train=df[0:-24]

    # order columns
    df_train=miscellaneous.order_columns(df_train,["ZONEID","timestamp","MONTH","DAY","HOUR","Forecasted Total Load","Forecasted Zonal Load","Zonal Price"])
    # save
    n=miscellaneous.get_task_number(file)
    df_train.to_csv(f"Data/Price/Task {n}/Task{n}_P_train.csv", index=False)

if __name__=="__main__":
    file=sys.argv[1]
    # clean passed file
    create_train(file)