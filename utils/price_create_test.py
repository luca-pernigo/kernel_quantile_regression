
import re
import pandas as pd
import sys

import holidays

import miscellaneous

# bash script
# for ((i=1;i<=15;i++)); do python utils/price_create_test.py "Data/Price/Task $i/Task${i}_P.csv"; done



def create_test(file):
    df=miscellaneous.clean_time(file, 2011, 2014)

    df_test=df[-24:]
    # print(df_test)

    # get new file containing prices we have to test for task i
    file_new=miscellaneous.get_test(file)
    df_new=pd.read_csv(f"{file_new}", sep=",", decimal=".")
        
    # join on df_test
    df_test.fillna(0)
    df_merged=df_test.merge(df_new, on="timestamp", how="left",suffixes=('_x', ''))
    
    df_merged.drop(df_merged.filter(regex='_x$').columns, axis=1, inplace=True)
    
    
    # order columns
    df_merged=miscellaneous.order_columns(df_merged,["ZONEID","timestamp","MONTH","DAY","HOUR","Forecasted Total Load","Forecasted Zonal Load","Zonal Price"])
    # save
    n=miscellaneous.get_task_number(file)
    
    
    if miscellaneous.get_task_number(file)==15:
        df_merged["ZONEID"]=df_new["ZONEID"]
        df_merged["Zonal Price"]=df_new["Zonal Price"]
        
    df_merged.to_csv(f"Data/Price/Task {n}/Task{n}_P_test.csv", index=False)


if __name__=="__main__":
    file=sys.argv[1]
    # clean passed file
    create_test(f"{file}")