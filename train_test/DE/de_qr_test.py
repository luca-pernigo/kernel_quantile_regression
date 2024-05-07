import matplotlib.pyplot as plt

import numpy as np

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


# load train test data
train=pd.read_csv(f"Data/DE/2021/clean/de.csv")
test=pd.read_csv(f"Data/DE/2022/clean/de.csv")

# X y
X_train=train[["Temperature","Wind_speed","Day_of_week","Is_holiday"]]
X_test=test[["Temperature","Wind_speed","Day_of_week","Is_holiday"]]

y_train=train["Load"]
y_test=test["Load"]

exp="full"
quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# predict df
df_predict=pd.DataFrame(columns=[f"{q}" for q in quantiles])

for i,q in enumerate(quantiles):
    # load model
    krn_q=pickle.load(open(f"train_test/DE/models_{exp}/krn_qr_{q}.pkl", "rb"))
    # predict
    y_predict_q=krn_q.predict(X_test_scaled)
    print(mean_pinball_loss(y_test,y_predict_q, alpha=q)/np.mean(y_test))
    
    # put in predict df
    df_predict[f"{q}"]=pd.Series(y_predict_q)


# save predictions to csv
df_predict.to_csv(f"Data/DE/2022/clean/model_prediction_{exp}.csv", index=False)


# compute pinball loss
pinball_tot=0
for i,q in enumerate(quantiles):
    # normalized pinball loss
    pinball_q=mean_pinball_loss(y_test,df_predict[f"{q}"], alpha=q)/np.mean(y_test)
    print(f"{q}  &  {pinball_q}  \\\\ ")
    # print(f"pinball loss quantile {q}: ", pinball_q)
    pinball_tot+=pinball_q


ans=pinball_tot/len(quantiles)
print("total quantile loss: ", ans)

