import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler

import sys


# script for plotting saved predictions

# select country
country="CH"
country_name="Switzerland"


df_pred=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/{country}/2022/clean/model_prediction.csv")
train=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/{country}/2021/clean/{country.lower()}.csv")
test=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/{country}/2022/clean/{country.lower()}.csv")

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# X y
X_train=train[["Temperature","Wind_speed"]]
X_test=test[["Temperature","Wind_speed"]]

y_train=train["Load"]
y_test=test["Load"]

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# confidence interval
# plot first predictions
n=100
plt.figure(figsize=(15,5))
x=np.linspace(0,len(y_test),num=len(y_test))[0:n]
plt.plot(x,y_test[0:n], color="black", label="effective")

sys.path.append("/Users/luca/Desktop/kernel_quantile_regression/")
# 95%
krn_q=pickle.load(open(f"/Users/luca/Desktop/kernel_quantile_regression/train_test/{country}/models/krn_qr_{0.95}.pkl", "rb"))
y_predict_95=krn_q.predict(X_test_scaled)
plt.fill_between(x,y_predict_95[0:n],y_test[0:n], alpha=0.4, color="green", edgecolor="red", label="90% Confidence interval")

# 5%
krn_q=pickle.load(open(f"train_test/{country}/models/krn_qr_{0.05}.pkl", "rb"))
y_predict_05=krn_q.predict(X_test_scaled)

plt.fill_between(x,y_test[0:n], y_predict_05[0:n], alpha=0.4, color="green", edgecolor="red")
plt.ylabel("Load (MW)")

plt.legend()
plt.xlabel("Observations")
plt.title(f"Probabilistic forecast for load in {country_name} (2022)")

# savefig
# plt.savefig(f"plots/{country}/{country}_load_CI.png")

plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# plot quantiles vs temperature and wind
plt.figure(figsize=(20,4))

cm_set1=plt.get_cmap('Set1')

plt.plot(test["Temperature"], test["Load"], "o", alpha=0.4, markerfacecolor="white", markeredgecolor="black")


for i,q in enumerate(quantiles):
    [plt.plot(test.sort_values(by='Temperature')["Temperature"], df_pred[f"{q}"], alpha=0.7, label=f"q={quantiles[i]}", color=cm_set1(i))]

plt.legend()

plt.ylabel("Load (MW)")
plt.xlabel("Temperature")
plt.title(f"Quantiles load versus temperature probabilistic forecast for load in {country_name} (2022)")

plt.savefig(f"plots/{country}/{country}_2022_quantiles.png")

plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# plot quanitles
plt.figure(figsize=(20,4))

cm_set1=plt.get_cmap('Set1')

for i,q in enumerate(quantiles):
    [plt.plot(df_pred[f"{q}"], alpha=0.7, label=f"q={quantiles[i]}", color=cm_set1(i))]


plt.legend()

plt.ylabel("Load (MW)")
plt.xlabel("Time")
plt.title(f"Quantiles probabilistic forecast for load in {country_name} (2022)")

plt.savefig(f"plots/{country}/{country}_load_vs_temperature_quantiles.png")

plt.show()