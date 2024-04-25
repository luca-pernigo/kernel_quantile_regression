# plot quantiles predictions DE

import matplotlib.pyplot as plt
import pandas as pd

import os
import sys

sys.path.append(os.getcwd())

from utils.miscellaneous import qs_plot

# n=2500
# read data
df_test=pd.read_csv("Data/CH/clean/ch_test.csv")
y_test=df_test["Load"]
# predictions
df=pd.read_csv("Data/CH/clean/model_prediction.csv")
df.rename(columns={"0.1":"0","0.2":"1","0.3":"2","0.4":"3","0.5":"4","0.6":"5","0.7":"6","0.8":"7","0.9":"8"}, inplace=True)

figsize = (20, 5)

# normalizing const
# nrm=df.sum(axis=1)

    
qs_plot(df, figsize, y_test)
plt.xlabel("Observations")
plt.ylabel("Load(MW)")
plt.title("Probabilistic forecast for load in Switzerland (2021)")


plt.savefig("plots/CH/CH_quantiles.png")
# plt.show()
