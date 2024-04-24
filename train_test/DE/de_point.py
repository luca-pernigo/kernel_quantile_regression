# script for point prediction metrics on median R2, RMSE, MAE

import numpy as np
import pandas as pd


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

test=pd.read_csv("Data/DE/2021/clean/de_test.csv")
y_test=test["Load"]

df_pred=pd.read_csv("Data/DE/2021/clean/model_prediction.csv")

pred=df_pred["0.5"]

print("R2 score: ", r2_score(y_test, pred))

print("Normalized RMSE: ", np.sqrt(mean_squared_error(y_test, pred))/np.mean(y_test))

print("Normalized MAE: ", mean_absolute_error(y_test, pred)/np.mean(y_test))


# \begin{table}[]
# \caption{CH}
# \begin{tabular}{lllll}
# Metric & Scores              &  &  &  \\
# R2     & 0.8160466760108515  &  &  &  \\
# NRMSE  & 0.07083200215373132 &  &  &  \\
# NMAE   & 0.05545222272925634 &  &  & 
# \end{tabular}
# \end{table}