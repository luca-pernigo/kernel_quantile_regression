# competitor performance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pprint import pprint

from quantile_forest import RandomForestQuantileRegressor as rfr

from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.metrics import mean_pinball_loss

import statsmodels.regression.quantile_regression as qr 
import sys

from tqdm import tqdm



if __name__=="__main__":
    country="CH"    
    # observed data
    hist=pd.read_csv(f"../../Data/{country}/2021/clean/{country.lower()}.csv")

    # load data
    df=pd.read_csv(f"../../Data/{country}/2022/clean/{country.lower()}.csv")
    df_len=len(df)
    
    time_window=24*1
    n_windows=df_len//time_window
    
    quantiles = [0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.95]
    # quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    qr_pinball_losses={q:0 for q in quantiles}
    gbm_pinball_losses={q:0 for q in quantiles}
    qf_pinball_losses={q:0 for q in quantiles}
    
    # predict df
    df_predict=pd.DataFrame(columns=[f"{q}" for q in quantiles])

    dict_predict={q:[] for q in quantiles}
    
    # rolling window
    for i in tqdm(range(0,df_len, time_window)):
        j=i+time_window
        z=j+time_window
        if j>df_len:
            j=df_len
            
        # train
        # create df train
        df_concat=pd.concat([hist, df.iloc[0:i,:]], ignore_index=True)
        df_sub=df_concat.iloc[-1500:,:]

        X_train=hist[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]]
        y_train=hist["Load"]
        m=len(y_train)
        
        # test
        X_test=df[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]].iloc[i:j,:]
        y_test=df["Load"].iloc[i:j]

        # fit all quantiles
        for i,q in enumerate(quantiles):
            # fit quantile q
            qr_=pickle.load(open(f"train_test/{country}/models_lqr_full/lqr_{int(q*10-1)}.pkl", "rb"))
            
            qr_gbr=pickle.load(open(f"train_test/{country}/models_gbm_qr_full/gbm_qr_{int(q*10-1)}.pkl", "rb"))
            
            qr_rf=pickle.load(open(f"train_test/{country}/models_qf_full/qf_{int(q*10-1)}.pkl", "rb"))

            # predict quantile q
            y_qr_predict_q=qr_.predict(X_test)
            qr_pinball_losses[q]+=1/n_windows*mean_pinball_loss(y_test,y_qr_predict_q, alpha=q)/np.mean(y_test)

            y_gbm_qr_predict_q=qr_gbr.predict(X_test)
            gbm_pinball_losses[q]+=1/n_windows*mean_pinball_loss(y_test,y_gbm_qr_predict_q, alpha=q)/np.mean(y_test)

            y_qf_predict_q=qr_rf.predict(X_test.values)
            qf_pinball_losses[q]+=1/n_windows*mean_pinball_loss(y_test,y_qf_predict_q, alpha=q)/np.mean(y_test)

pd.set_option('display.max_columns', 11)

qr_df_pinball=pd.DataFrame(data=qr_pinball_losses, index=[1])
print("Linear",qr_df_pinball)
qr_df_pinball.to_csv(f"../../Data/{country}/2022/clean/rolling_window/pinball_rolling_window_qr.csv", index=False)

gbm_qr_df_pinball=pd.DataFrame(data=gbm_pinball_losses, index=[1])
print("Gradient boosting machine",gbm_qr_df_pinball)
gbm_qr_df_pinball.to_csv(f"../../Data/{country}/2022/clean/rolling_window/pinball_rolling_window_gbm_qr.csv", index=False)

qf_qr_df_pinball=pd.DataFrame(data=qf_pinball_losses, index=[1])
print("Quantile forest",qf_qr_df_pinball)
qf_qr_df_pinball.to_csv(f"../../Data/{country}/2022/clean/rolling_window/pinball_rolling_window_qf_qr.csv", index=False)
