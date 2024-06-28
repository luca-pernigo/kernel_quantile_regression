# script to train models on cleaned data CH
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler
import sys
from tqdm import tqdm

from kernel_quantile_regression.kqr import KQR

import time

country="CH"
ktype="a_laplacian"

# observed data
hist=pd.read_csv(f"Data/{country}/2021/clean/{country.lower()}.csv")

# load data
# df=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/SECURES-Met/{country}/clean/test/2021/df.csv")
df=pd.read_csv(f"Data/{country}/2022/clean/{country.lower()}.csv")
df_len=len(df)

time_window=24*1
n_windows=df_len//time_window
# quantiles
# quantiles = [i/100 for i in range(1,100)]
quantiles = [0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.95]
# quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

pinball_losses={q:0 for q in quantiles}
# predict df
df_predict=pd.DataFrame(columns=[f"{q}" for q in quantiles])

dict_predict={q:[] for q in quantiles}

# rolling window
for i in tqdm(range(0,df_len, time_window)):
    j=i+time_window
    z=j+time_window
    if j>df_len:
        j=df_len
        # j=-1

        # break
    # if z>df_len:
    #     z=df_len

    # train
    # create df train
    df_concat=pd.concat([hist, df.iloc[0:i,:]], ignore_index=True)
    df_sub=df_concat.iloc[-1500:,:]

    # X_train=df[["Direct_irradiation","Global_radiation","Hydro_reservoir","Hydro_river","Temperature","Wind_potential"]].iloc[i:j,:]
    X_train=hist[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]]
    y_train=hist["Load"]
    m=len(y_train)
    
    # test
    X_test=df[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]].iloc[i:j,:]
    y_test=df["Load"].iloc[i:j]

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # fit all quantiles
    for i,q in enumerate(quantiles):
        # fit quantile q
        krn_q=pickle.load(open(f"train_test/{country}/models_{ktype}/krn_qr_{q}.pkl", "rb"))

        # predict quantile q
        y_predict_q=krn_q.predict(X_test_scaled)
        pinball_losses[q]+=1/n_windows*mean_pinball_loss(y_test,y_predict_q, alpha=q)/np.mean(y_test)

        # print(y_predict_q)
        dict_predict[q]+=y_predict_q.tolist()
        
        # time.sleep(5)

pd.set_option('display.max_columns', 11)

df_predict=pd.DataFrame(data=dict_predict)
# save predictions to csv
# df_predict.to_csv(f"Data/SECURES-Met/{country}/clean/model_prediction_rolling_window_{ktype}.csv", index=False)
df_predict.to_csv(f"Data/{country}/2022/clean/rolling_window/model_prediction_rolling_window_{ktype}.csv", index=False)


df_pinball=pd.DataFrame(data=pinball_losses, index=[1])
print(df_pinball)
df_pinball.to_csv(f"Data/{country}/2022/clean/rolling_window/pinball_rolling_window_{ktype}.csv", index=False)