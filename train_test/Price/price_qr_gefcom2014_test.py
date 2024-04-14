
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler
import sys

from tqdm import tqdm
import os

wd = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(wd, '..', '..'))
sys.path.append(project_directory)

from utils import miscellaneous

def test(ith):

    task_month={4:7,5:7,6:7,7:7,8:7,9:7,10:7,11:7,12:7,13:12,14:12,15:12}

    df=pd.read_csv(f"Data/Price/Task {ith}/Task{ith}_P_test.csv")

    X_test=df[["DAY","HOUR","Forecasted Total Load","Forecasted Zonal Load"]]
    y_test=df["Zonal Price"]

    quantiles = [i/100 for i in range(1,100)]

    # we need to scale X_test
    df_train=pd.read_csv(f"Data/Price/Task {ith}/Task{ith}_P_train.csv")
    df_train=df_train[(df_train["MONTH"]==task_month[ith])]

    X_train=df_train[["DAY","HOUR","Forecasted Total Load","Forecasted Zonal Load"]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pinball_tot=0

    # predict
    df_template_submission=pd.read_csv(f"Data/Price/Task {ith}/Benchmark{ith}_P.csv")

    df_predict=df_template_submission[["ZONEID", "timestamp"]].copy()

    for i,q in enumerate(quantiles):
        krn_q=pickle.load(open(f"train_test/Price/models/task {ith}/krn_qr_{i}.pkl", "rb"))
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