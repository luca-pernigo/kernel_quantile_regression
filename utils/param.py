# script to print the saved best hyperparameters
from quantile_forest import RandomForestQuantileRegressor as rfr

from sklearn.ensemble import GradientBoostingRegressor as gbr

import pickle

from kernel_quantile_regression.kqr import KQR

# gbm_qr=pickle.load(open("/Users/luca/Desktop/kernel_quantile_regression/train_test/CH/models_gbm_qr_full/gbm_qr_0.pkl", "rb"))
# print(gbm_qr)

# qf_qr=pickle.load(open("/Users/luca/Desktop/kernel_quantile_regression/train_test/DE/models_gbm_qr_full/gbm_qr_0.pkl", "rb"))
# print(qf_qr)

kqr=pickle.load(open("/Users/luca/Desktop/kernel_quantile_regression/train_test/DE/models_a_laplacian/krn_qr_0.7.pkl", "rb"))
print(kqr.gamma,kqr.C)