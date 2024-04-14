import pandas as pd

# step 1 create clean test
import pandas as pd
import sys

import holidays

import miscellaneous

# for ((i=1;i<=15;i++)); do python utils/load_create_test.py "Data/Load/Task $i//L{ith}-test.csv"; done

def create_test(file):
    ith=miscellaneous.get_task_number(file)
    if ith==15:
        df=pd.read_csv(f"Data/Load/Task {ith}/L{ith}-test.csv")
        # dataset 15 does not stick to convention used so far in the competition, thus we have an if conditional to handle it
        df['TIMESTAMP'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
        df=miscellaneous.datetime_conv(df, "TIMESTAMP")
        # to be coherent hours go from 0 to 23
        df['HOUR'] =df["hour"]-1


        # drop not useful columns
        df.drop(columns=['TIMESTAMP', 'date', 'hour'], inplace=True)

        # average temperatures
        weather_stat=df.filter(regex=("w.*")).columns.to_list()
        # average them 
        df['w_avg'] = df[weather_stat].mean(axis=1)

        # drop columns
        df.drop(columns=weather_stat, inplace=True)
        df.drop(columns=["YEAR"], inplace=True)

        df=df[["LOAD","DAY","MONTH","HOUR","DAY_OF_WEEK","IS_HOLIDAY","w_avg"]]
        
        # save to csv
        df_test=df

        # print(df)

    else:    
        df=pd.read_csv(f"Data/Load/Task {ith}/L{ith}-test.csv")

        df=miscellaneous.datetime_conv(df, "TIMESTAMP")

        # drop not useful columns
        df.drop(columns=['ZONEID', 'TIMESTAMP'], inplace=True)


        # average temperatures
        weather_stat=df.filter(regex=("w.*")).columns.to_list()
        # average them 
        df['w_avg'] = df[weather_stat].mean(axis=1)

        # drop columns
        df.drop(columns=weather_stat, inplace=True)
        df.drop(columns=["YEAR"], inplace=True)

        df_test=df

    df_w=pd.read_csv(f"Data/Load/L-weather-train.csv")

    # drop leap year
    df_w=df_w[~((df_w["MONTH"]==2) & (df_w["DAY"]==29))]

    w_avg=df_w.groupby(["MONTH","DAY","HOUR"]).mean("w_avg").reset_index()


    # print(len(w_avg))
    task_month={1:10,2:11,3:12,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8,12:9,13:10,14:11,15:12}

    m=task_month[ith]
        
    w_avg_month=w_avg[w_avg["MONTH"]==m]
    
    if m!=12:
        idx_next=w_avg_month.index[-1]
    else:
        idx_next=0
    # print(end)
    # print(w_avg.iloc[end-1])
    
    # add last row because test data ends at 00:00 of the next month
    w_avg_month.loc[len(w_avg_month)] = w_avg.iloc[idx_next+1]
    
    # drop first row because test data starts from hour 01:00, thus drop data for hour 00:00
    w_avg_month.drop(index=w_avg_month.index[0], axis=0, inplace=True)
    # print(w_avg_month)
    
    df_test["w_avg"]=w_avg_month["w_avg"].values

    df_test.to_csv(f"Data/Load/Task {ith}/L{ith}-test_clean.csv", index=False)


if __name__=="__main__":
   file=sys.argv[1]
   create_test(sys.argv[1])


