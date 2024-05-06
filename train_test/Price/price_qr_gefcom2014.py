import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from tqdm import tqdm

from kernel_quantile_regression.kqr import KQR



if __name__=="__main__":
    ith=int(sys.argv[1])
    
    
    task_month={4:7,5:7,6:7,7:7,8:7,9:7,10:7,11:7,12:7,13:12,14:12,15:12}
    # load data
    df_train=pd.read_csv(f"Data/Price/Task {ith}/Task{ith}_P_train.csv")
    df_train=df_train[(df_train["MONTH"]==task_month[ith])][-1440:]
    # define train
    X_train=df_train[["DAY","HOUR","Forecasted Total Load","Forecasted Zonal Load"]]
    y_train=df_train["Zonal Price"]

    # 99 quantiles
    quantiles = [i/100 for i in range(1,100)]

    # kernel quantile regression
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    qr_krn_models=[]
    y_test_pred_qr_krn=[]

    param_grid_krn = dict(
    C=[1e-1,1e-2,1, 5, 10,1e2,1e4],
    gamma=[1e-1,1e-2,0.5,1, 5, 10, 20]   
    )
    
    ktype="laplacian"
    for i,q in enumerate(tqdm(quantiles)):
        
        # define loss to tune
        # greater_is_better=False, we need to maximize the negative of the loss
        neg_mean_pinball_loss_scorer = make_scorer(
        mean_pinball_loss,
        alpha=q,
        greater_is_better=False, 
        )

        krn_blueprint=KQR(alpha=q)
        best_hyperparameters_krn=HalvingRandomSearchCV(
                krn_blueprint,
                param_grid_krn,
                scoring=neg_mean_pinball_loss_scorer,
                n_jobs=2,
                random_state=0,
            ).fit(X_train_scaled, y_train).best_params_

        # fit data for specific quantile
        qr_krn_models+=[KQR(alpha=q, **best_hyperparameters_krn, kernel_type=ktype).fit(X_train_scaled, y_train)]

        # save models to pickle
        pickle.dump(qr_krn_models[i], open(f'train_test/Price/{ktype}/task {ith}/krn_qr_{i}.pkl', 'wb'))
        
