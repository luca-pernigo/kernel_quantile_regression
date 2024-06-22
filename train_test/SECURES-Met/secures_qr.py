# script to train models on SECURES-Met data
import itertools

import math
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV

from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from tqdm import tqdm

from kernel_quantile_regression.kqr import KQR

if __name__=="__main__":

    country=sys.argv[1]
    ktype=sys.argv[2]
    # load train data
    df=pd.read_csv(f"Data/SECURES-Met/{country}/clean/train/df.csv")

    # quantiles
    # quantiles = [i/100 for i in range(1,100)]
    # quantiles = [0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.95]
    # quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    quantiles = [0.05, 0.95]


    # for the experiment with just temperature and wind_speed exp=""
    # X y
    X_train=df[["Direct_irradiation","Global_radiation","Hydro_reservoir","Hydro_river","Temperature","Wind_potential"]]
    y_train=df["Load"]

    # scale
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # kernel quantile regression
    qr_krn_models=[]
    y_test_pred_qr_krn=[]

    m=len(y_train)
    
    # this to avoid cross validating again, because we already know best hyperparameters
    # we just have to fit a new model with different quantiles
    # krn_q=pickle.load(open(f'/Users/luca/Desktop/ThesisKernelMethods/experiments/train_test/SECURES-Met/{country}/{ktype}/krn_qr_{0.5}.pkl', 'rb'))
    # best_gamma=krn_q.C
    # best_C=krn_q.gamma

    # param_grid_krn = dict(
    # d= [2,3],
    # gamma=[1,2,3]
    # )

    # neg_mean_pinball_loss_scorer_05 = make_scorer(
    #     mean_pinball_loss,
    #     alpha=0.5,
    #     greater_is_better=False,
    #     )

    # krn_blueprint=KQR(alpha=0.5, C=1,c=0, kernel_type=ktype)
    # cv=HalvingGridSearchCV(
    #         krn_blueprint,
    #         param_grid_krn,
    #         scoring=neg_mean_pinball_loss_scorer_05,
    #         n_jobs=2,
    #         random_state=0,
    #     ).fit(X_train, y_train)
    # best_hyperparameters_krn=cv.best_params_

    # # cv results
    # df_cv_res=pd.DataFrame(cv.cv_results_)
    # df_cv_res.to_csv(f"train_test/SECURES-Met/{country}/{ktype}/models_{ktype}_gridsearch.csv",index=False)

    # train
    for i,q in enumerate(tqdm(quantiles)):
        # print(best_hyperparameters_krn)
        # fit data for specific quantile
        qr_krn_models+=[KQR(alpha=q, gamma=8,C=1, kernel_type=ktype).fit(X_train_scaled, y_train)]

        # save models to pickle
        pickle.dump(qr_krn_models[i], open(f'train_test/SECURES-Met/{country}/{ktype}/krn_qr_{q}.pkl', 'wb'))
        


# parameters for kernel comparison with C=1
# materns gamma= [1,2,4,8]
# linear, cosine none
# polynomial d= [2,3], gamma=[1,2,3], ch crossvalidated by hand because cvxopt infeasibility makes code stop for some combinations
# periodic p=24,gamma=[1,2,4,8]
# sigmoid c=0, gamma=[350,400,1000] smaller gammas break