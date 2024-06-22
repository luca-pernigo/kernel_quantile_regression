import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
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
    
    ktype=sys.argv[2]
    # load data
    df=pd.read_csv("Data/Load/L-train.csv")
    print("tot_data", len(df))
    # in the load track each task predicts the month=task number -3
    df=df[df["MONTH"]==(ith-3)][-2232:]
    print("month data", len(df))

    X_train=df[["DAY",  "HOUR",  "DAY_OF_WEEK",  "IS_HOLIDAY",  "w_avg"]]
    y_train=df["LOAD"]

    m=len(y_train)
    # 99 quantiles
    quantiles = [i/100 for i in range(1,100)]

    # kernel quantile regression
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    qr_krn_models=[]
    y_test_pred_qr_krn=[]

    param_grid_krn = dict(
    C=[1/(m*10e-6),1/(m*10e-5),1/(m*10e-4),1/(m*10e-3),1/(m*10e-2),1/(m*10e-1),1/(m*10e-0),1/(m*10e1)],
    gamma=[2**-6,2**-5,2**-4, 2**-3,2**-2,2**-1,1, 2, 2**2, 2**3,2**4,2**5,2**6]   
    )    

    # define loss to tune
    neg_mean_pinball_loss_scorer = make_scorer(
    mean_pinball_loss,
    alpha=0.5,
    greater_is_better=False,
    )

    krn_blueprint=KQR(alpha=0.5, kernel_type=ktype)
    cv=GridSearchCV(
            krn_blueprint,
            param_grid_krn,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=2
        ).fit(X_train_scaled, y_train)
    
    best_hyperparameters_krn=cv.best_params_
    
    for i,q in enumerate(tqdm(quantiles)):
        # print(best_hyperparameters_krn)

        # fit data for specific quantile
        qr_krn_models+=[KQR(alpha=q, **best_hyperparameters_krn, kernel_type=ktype).fit(X_train_scaled, y_train)]

        # save models to pickle
        pickle.dump(qr_krn_models[i], open(f'train_test/Load/{ktype}/task {ith}/krn_qr_{i}.pkl', 'wb'))

# cv results
df_cv_res=pd.DataFrame(cv.cv_results_)
df_cv_res.to_csv(f"train_test/Load/{ktype}/task {ith}/models_{ktype}_gridsearch.csv",index=False)