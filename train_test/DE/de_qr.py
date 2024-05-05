# script to train models on cleaned data
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

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
# quantiles = [0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.95]
quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

exp="full"
# X y
X_train=df[["Temperature","Wind_speed","Day_of_week","Is_holiday"]]
y_train=df["Load"]

# scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# kernel quantile regression
qr_krn_models=[]
y_test_pred_qr_krn=[]

# param_grid_krn = dict(
# C=[1e-1,1e-2,1, 5, 10,1e2,1e4],
# gamma=[1e-1,1e-2,0.5,1, 5, 10, 20]   
# )
ktype="laplacian"

neg_mean_pinball_loss_scorer_05 = make_scorer(
    mean_pinball_loss,
    alpha=0.5,
    greater_is_better=False,
    )
# krn_blueprint=KQR(alpha=0.5)
# best_hyperparameters_krn=HalvingRandomSearchCV(
#         krn_blueprint,
#         param_grid_krn,
#         scoring=neg_mean_pinball_loss_scorer_05,
#         n_jobs=2,
#         random_state=0,
#     ).fit(X_train, y_train).best_params_
    
# train
for i,q in enumerate(tqdm(quantiles)):
    # fit data for specific quantile
    qr_krn_models+=[KQR(alpha=q, C=1300, gamma=0.1, kernel_type=ktype).fit(X_train_scaled, y_train)]

    # save models to pickle
    pickle.dump(qr_krn_models[i], open(f'train_test/DE/models_{exp}/krn_qr_{q}.pkl', 'wb'))
    
