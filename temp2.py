import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

country="CH"
ktype="a_laplacian"

# df_predict=pd.read_csv(f"Data/SECURES-Met/{country}/clean/model_prediction_rolling_window_{ktype}.csv", sep=',')
df_predict=pd.read_csv(f"Data/CH/2022/clean/model_prediction_rolling_window_{ktype}.csv", sep=',')

# y_test=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/SECURES-Met/{country}/clean/test/2021/df.csv")["Load"][24*7*1+1:].values
y_test=pd.read_csv(f"Data/CH/2022/clean/ch.csv")["Load"].values

l=2000
u=3000
plt.figure(figsize=(150,5))
x=np.linspace(1,u-l+1,u-l)
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
plt.show()

print(len(y_test), len(y_predict_05), len(y_predict_95))

