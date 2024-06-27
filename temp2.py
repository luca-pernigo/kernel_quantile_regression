import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

country="CH"
ktype="a_laplacian"

# df_predict=pd.read_csv(f"Data/SECURES-Met/{country}/clean/model_prediction_rolling_window_{ktype}.csv", sep=',')
df_predict=pd.read_csv(f"Data/CH/2022/clean/model_prediction_rolling_window_{ktype}.csv", sep=',')

# y_test=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/SECURES-Met/{country}/clean/test/2021/df.csv")["Load"][24*7*1+1:].values
y_test=pd.read_csv(f"Data/CH/2022/clean/ch.csv")["Load"].values


plt.plot(y_test, color="black", label="effective")

# 95%
y_predict_9=df_predict.loc[:,"0.9"].values
plt.plot(y_predict_9, alpha=0.4, color="red")

# 5%

y_predict_0=df_predict.loc[:,"0.1"].values

plt.plot(y_predict_0, alpha=0.4, color="blue")
plt.ylabel("Load (MW)")

plt.legend()
plt.xlabel("Observations")
# plt.title(f"Probabilistic forecast for load in {country_name} (2022)")
plt.show()

