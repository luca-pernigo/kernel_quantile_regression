import pandas as pd

df_w=pd.read_csv(f"Data/Load/L-weather-train.csv")

# drop leap year
df_w=df_w[~((df_w["MONTH"]==2) & (df_w["DAY"]==29))]

w_avg=df_w.groupby(["MONTH","DAY","HOUR"]).mean("w_avg").reset_index()


# print(len(w_avg))
task_month={1:10,2:11,3:12,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8,12:9,13:10,14:11,15:12}

for t,m in task_month.items():
    
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

    df_test=pd.read_csv(f"Data/Load/Task {t}/L{t}-test_clean.csv")
    
    df_test["w_avg"]=w_avg_month["w_avg"].values

    df_test.to_csv(f"/Users/luca/Desktop/GEFCom2014 Data/Load/Task {t}/L{t}-avg_test_clean.csv", index=False)


