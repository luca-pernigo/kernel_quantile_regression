
import pandas as pd
import pickle

import os

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler
import sys

from tqdm import tqdm

wd = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(wd, '..', '..'))
sys.path.append(project_directory)

from utils import miscellaneous



def test(ith):
    df=pd.read_csv(f"Data/Load/Task {ith}/L{ith}-test_clean.csv")

    X_test=df[["DAY", "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    y_test=df["LOAD"]

    quantiles = [i/100 for i in range(1,100)]

    # we need to scale X_test
    df_train=pd.read_csv("Data/Load/L-train.csv")
    df_train=df_train[df_train["MONTH"]==(ith-3)]
    X_train=df_train[["DAY",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pinball_tot=0
    
    # predict
    df_template_submission=pd.read_csv(f"Data/Load/Task {ith}/L{ith}-benchmark.csv")

    df_predict=df_template_submission[["ZONEID", "TIMESTAMP"]].copy()

    for i,q in enumerate(quantiles):
        krn_q=pickle.load(open(f"train_test/Load/models/task {ith}/krn_qr_{i}.pkl", "rb"))
        y_predict_q=krn_q.predict(X_test_scaled)
        
        df_predict[f"{q}"]=pd.Series(y_predict_q)

    # reorder quantiles
    res=miscellaneous.order_quantiles(df_predict)

    # compute pinball loss
    pinball_tot=0
    for i,q in enumerate(quantiles):
        predict=res.iloc[:,i]
        pinball_q=mean_pinball_loss(y_test,predict, alpha=q)
        print(f"pinball loss quantile {q}: ", pinball_q)
        pinball_tot+=pinball_q

    ans=pinball_tot/len(quantiles)
    print("total quantile: ", ans)
    
    return ans

if __name__=="__main__":
    i=int(sys.argv[1])
    ans=test(i)


    # original_stdout=sys.stdout
    # with open(f"/Users/luca/Desktop/ThesisKernelMethods/thesis/tables/qr_gefcom2014.txt", "a") as f:
    #     sys.stdout=f
    #     # print(f"Average pinball loss, task n. {i}: ")
    #     print(f"{ans:.4f}")
    #     print("&")

    #     sys.stdout=original_stdout