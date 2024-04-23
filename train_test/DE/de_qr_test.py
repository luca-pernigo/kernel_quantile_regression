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
train=pd.read_csv(f"Data/DE/2021/clean/de_train.csv")
test=pd.read_csv(f"Data/DE/2021/clean/de_test.csv")

# X y
X_train=train[["Temperature","Day","Month","Hour","Day_of_week","Is_holiday"]]
X_test=test[["Temperature","Day","Month","Hour","Day_of_week","Is_holiday"]]

y_train=train["Load"]
y_test=test["Load"]

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1))

# predict df
df_predict=pd.DataFrame(columns=[f"{q}" for q in quantiles])

for i,q in enumerate(quantiles):
    # load model
    krn_q=pickle.load(open(f"train_test/DE/models/krn_qr_{q}.pkl", "rb"))
    # predict
    y_predict_q=krn_q.predict(X_test_scaled)
    
    # put in predict df
    df_predict[f"{q}"]=pd.Series(scaler_y.inverse_transform(y_predict_q.reshape(-1,1)).reshape(-1,))


# plot first predictions
n=500
plt.figure(figsize=(15,5))
x=np.linspace(0,len(y_test_scaled),num=len(y_test_scaled))[0:n]
plt.plot(x,y_test[0:n], color="black")

# 95%
krn_q=pickle.load(open(f"train_test/DE/models/krn_qr_{0.95}.pkl", "rb"))
y_predict_95=krn_q.predict(X_test_scaled)
plt.fill_between(x,scaler_y.inverse_transform(y_predict_95.reshape(-1,1)).ravel()[0:n],y_test[0:n], alpha=0.4, color="green", edgecolor="red")

# 5%
krn_q=pickle.load(open(f"train_test/DE/models/krn_qr_{0.05}.pkl", "rb"))
y_predict_05=krn_q.predict(X_test_scaled)

plt.fill_between(x,y_test[0:n], scaler_y.inverse_transform(y_predict_05.reshape(-1,1)).ravel()[0:n], alpha=0.4, color="green", edgecolor="red")

plt.show()

# save predictions to csv
df_predict.to_csv(f"Data/DE/2021/clean/model_prediction.csv", index=False)


# compute pinball loss
pinball_tot=0
for i,q in enumerate(quantiles):
    # normalized pinball loss
    pinball_q=mean_pinball_loss(y_test,df_predict[f"{q}"], alpha=q)/np.mean(y_test)
    print(f"pinball loss quantile {q}: ", pinball_q)
    pinball_tot+=pinball_q


ans=pinball_tot/len(quantiles)
print("total quantile loss: ", ans)

