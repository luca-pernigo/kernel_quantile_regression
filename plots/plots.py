import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import pickle
from utils import ktype_title, plot_ci
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('src/kernel_quantile_regression/')

from kernel_quantile_regression.kqr import KQR
# script for plotting saved predictions

# select options

# select country
country="DE"
country_name="Germany"

ktype="a_laplacian"

l=1000
u=3000

year="2022"

# load data
df_pred=pd.read_csv(f"Data/{country}/2022/clean/model_prediction_{ktype}.csv")
train=pd.read_csv(f"Data/{country}/2021/clean/{country.lower()}.csv")
test=pd.read_csv(f"Data/{country}/2022/clean/{country.lower()}.csv")

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# X y
X_train=train[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]]
X_test=test[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]]

y_train=train["Load"]
y_test=test["Load"]

# scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 95%
krn_q=pickle.load(open(f"train_test/{country}/models_{ktype}/krn_qr_{0.95}.pkl", "rb"))
y_predict_95=krn_q.predict(X_test_scaled)

# # 5%
krn_q=pickle.load(open(f"train_test/{country}/models_{ktype}/krn_qr_{0.05}.pkl", "rb"))
y_predict_05=krn_q.predict(X_test_scaled)

plot_ci(y_test,y_predict_95,y_predict_05,df_pred,u,l,country_name,year)
# savefig
plt.savefig(f"plots/{country}/{country}_load_CI.png")
# plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # plot SECURES-Met
# year=2021

# df_pred=pd.read_csv(f"Data/SECURES-Met/{country}/clean/model_prediction_{ktype}.csv")
# train=pd.read_csv(f"Data/SECURES-Met/{country}/clean/train/df.csv")
# test=pd.read_csv(f"Data/SECURES-Met/{country}/clean/test/{year}/df.csv")

# quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# # X y
# X_train=train[["Direct_irradiation","Global_radiation","Hydro_reservoir","Hydro_river","Temperature","Wind_potential"]]
# X_test=test[["Direct_irradiation","Global_radiation","Hydro_reservoir","Hydro_river","Temperature","Wind_potential"]]

# y_train=train["Load"]
# y_test=test["Load"]

# # scale data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # confidence interval
# # plot a subset to avoid visual clutter
# l=6000
# u=7000
# plt.figure(figsize=(20,5))

# # 95%
# krn_q=pickle.load(open(f"train_test/SECURES-Met/{country}/{ktype}/krn_qr_{0.95}.pkl", "rb"))
# y_predict_95=krn_q.predict(X_test_scaled)

# # 5%
# krn_q=pickle.load(open(f"train_test/SECURES-Met/{country}/{ktype}/krn_qr_{0.05}.pkl", "rb"))
# y_predict_05=krn_q.predict(X_test_scaled)

# plot_ci(y_test,y_predict_95,y_predict_05,df_pred,u,l,country_name,year)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # plot quantiles vs temperature and wind
# # plt.figure(figsize=(20,4))

# # cm_set1=plt.get_cmap('Set1')

# # plt.plot(test["Temperature"], test["Load"], "o", alpha=0.4, markerfacecolor="white", markeredgecolor="black")


# # for i,q in enumerate(quantiles):
# #     [plt.plot(test.sort_values(by='Temperature')["Temperature"], df_pred[f"{q}"], alpha=0.7, label=f"q={quantiles[i]}", color=cm_set1(i))]

# # plt.legend()

# # plt.ylabel("Load (MW)")
# # plt.xlabel("Temperature")
# # plt.title(f"Quantiles load versus temperature probabilistic forecast for load in {country_name} (2022)")

# # plt.savefig(f"plots/{country}/{country}_2022_quantiles.png")

# # plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # plot quanitles
# # plt.figure(figsize=(20,4))

# # cm_set1=plt.get_cmap('Set1')

# # for i,q in enumerate(quantiles):
# #     [plt.plot(df_pred[f"{q}"], alpha=0.7, label=f"q={quantiles[i]}", color=cm_set1(i))]


# # plt.legend()

# # plt.ylabel("Load (MW)")
# # plt.xlabel("Time")
# # plt.title(f"Quantiles probabilistic forecast for load in {country_name} (2022)")

# # plt.savefig(f"plots/{country}/{country}_load_vs_temperature_quantiles.png")

# # plt.show()