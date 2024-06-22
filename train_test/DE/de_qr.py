# script to train models on cleaned data
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, GridSearchCV

from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from tqdm import tqdm

from kernel_quantile_regression.kqr import KQR

# load data
df=pd.read_csv("Data/DE/2021/clean/de.csv")

# quantiles
# quantiles = [i/100 for i in range(1,100)]
# quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
quantiles = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]

exp="full"
# X y
X_train=df[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]]
y_train=df["Load"]

m=len(y_train)
# scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# kernel quantile regression
qr_krn_models=[]
y_test_pred_qr_krn=[]

ktype="a_laplacian"

# param_grid_krn = dict(
# C=[1/(m*10e-6),1/(m*10e-5),1/(m*10e-4),1/(m*10e-3),1/(m*10e-2),1/(m*10e-1),1/(m*10e-0),1/(m*10e1)],
# gamma=[2**-6,2**-5,2**-4, 2**-3,2**-2,2**-1,1, 2, 2**2, 2**3,2**4,2**5,2**6]   
# )

# neg_mean_pinball_loss_scorer_05 = make_scorer(
#     mean_pinball_loss,
#     alpha=0.5,
#     greater_is_better=False,
#     )
# krn_blueprint=KQR(alpha=0.5)

# cv=GridSearchCV(
#         krn_blueprint,
#         param_grid_krn,
#         scoring=neg_mean_pinball_loss_scorer_05,
#         n_jobs=2
#     ).fit(X_train, y_train)

# best_hyperparameters_krn=cv.best_params_

# # cv results
# df_cv_res=pd.DataFrame(cv.cv_results_)
# df_cv_res.to_csv(f"train_test/DE/models_{ktype}/gridsearch.csv",index=False)

# train
for i,q in enumerate(tqdm(quantiles)):
    # fit data for specific quantile
    qr_krn_models+=[KQR(alpha=q, gamma=4, C=1/(m*1e-5), kernel_type=ktype).fit(X_train_scaled, y_train)]

    # save models to pickle
    pickle.dump(qr_krn_models[i], open(f'train_test/DE/models_{ktype}/krn_qr_{q}.pkl', 'wb'))

    
