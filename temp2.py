import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

country="CH"
ktype="a_laplacian"

# df_predict=pd.read_csv(f"Data/SECURES-Met/{country}/clean/model_prediction_rolling_window_{ktype}.csv", sep=',')
df_predict=pd.read_csv(f"Data/CH/2022/clean/model_prediction_rolling_window_{ktype}.csv", sep=',')

# y_test=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/SECURES-Met/{country}/clean/test/2021/df.csv")["Load"][24*7*1+1:].values
y_test=pd.read_csv(f"Data/CH/2022/clean/ch.csv")["Load"].values

dates=pd.date_range('2022-01-01', periods=len(df_predict), freq='H')
dates=pd.Series(dates)
l=0
u=len(df_predict)
plt.figure(figsize=(150,5))
x=np.linspace(0,u-l,u-l)

x_position=dates[ (dates.dt.hour==0) & (dates.dt.day==1)].index.values

u_pos = np.absolute(x_position[x_position<=u]-u)
u_idx=u_pos.argmin()


l_pos = np.absolute(x_position[x_position<=l]-l)
l_idx=l_pos.argmin()


months=["Genuary","February","March","April","May","June","July","August","September","October","November","December"]
# print(l_pos,l_idx,months[:l_idx+1],x_position[:l_idx+1])

plt.plot(x,y_test[l:u], color="black", label="effective")


plt.plot(x, y_test[l:u], color="black", label="effective")

# 95%
y_predict_95=df_predict.loc[:,"0.95"].values
# plt.plot(y_predict_95, alpha=0.4, color="red")

plt.fill_between(x,y_predict_95[l:u],y_test[l:u],where=y_predict_95[l:u]>=y_test[l:u], alpha=0.4, color="green", edgecolor="black", label="90% Confidence interval")
plt.fill_between(x,y_predict_95[l:u],y_test[l:u],where=y_predict_95[l:u]<y_test[l:u], alpha=0.4, color="red", edgecolor="black", label="90% Confidence interval")



# 5%

y_predict_05=df_predict.loc[:,"0.05"].values

# plt.plot(y_predict_05, alpha=0.4, color="blue")
plt.fill_between(x,y_test[l:u], y_predict_05[l:u], where=(y_predict_05[l:u]<=y_test[l:u]) & (y_predict_95[l:u]>=y_test[l:u]), alpha=0.4, color="green", edgecolor="black")
plt.fill_between(x,y_test[l:u], y_predict_05[l:u], where=y_predict_05[l:u]>=y_test[l:u], alpha=0.4, color="red", edgecolor="black")

plt.ylabel("Load (MW)")

plt.legend()
plt.xlabel("Observations")
# plt.title(f"Probabilistic forecast for load in {country_name} (2022)")
plt.xticks(x_position[l_idx:u_idx+1]-l, months[l_idx:u_idx+1])
plt.show()

print(len(y_test), len(y_predict_05), len(y_predict_95))

