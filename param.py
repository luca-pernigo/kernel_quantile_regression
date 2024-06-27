from quantile_forest import RandomForestQuantileRegressor as rfr

from sklearn.ensemble import GradientBoostingRegressor as gbr

import pickle

# gbm_qr=pickle.load(open("/Users/luca/Desktop/kernel_quantile_regression/train_test/CH/models_gbm_qr_full/gbm_qr_0.pkl", "rb"))
# print(gbm_qr)

qf_qr=pickle.load(open("/Users/luca/Desktop/kernel_quantile_regression/train_test/CH/models_qf_full/qf_0.pkl", "rb"))
print(qf_qr)
