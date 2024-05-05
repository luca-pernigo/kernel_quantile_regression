# competitor performance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pprint import pprint

from quantile_forest import RandomForestQuantileRegressor as rfr

from sklearn.ensemble import GradientBoostingRegressor as gbr

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

import statsmodels.regression.quantile_regression as qr 
import sys

from tqdm import tqdm



if __name__=="__main__":
    
    # script for comparing quantile regressors algorithms
    # select country
    country="DE"
    exp="full"
    
    # quantiles
    quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    # load data
    train=pd.read_csv(f"Data/{country}/2021/clean/{country.lower()}.csv")
    test=pd.read_csv(f"Data/{country}/2022/clean/{country.lower()}.csv")
    pinball_scores=pd.read_csv(f"train_test/{country}/scores_{exp}.csv")
    
    # X y
    X_train=train[["Temperature","Wind_speed","Day_of_week","Is_holiday"]]
    X_test=test[["Temperature","Wind_speed","Day_of_week","Is_holiday"]]

    y_train=train["Load"]
    y_test=test["Load"]

   
    # linear quantile regression
    qr_models = [qr.QuantReg(y_train, X_train).fit(q=q) for q in quantiles]
    
    y_test_pred_qr=[qr_model.predict(X_test) for qr_model in qr_models]
    

    # save 
    [pickle.dump(qr_models[i], open(f"/Users/luca/Desktop/kernel_quantile_regression/train_test/{country}/models_lqr_{exp}/lqr_{i}", "wb")) for i in range(9)]
    
    
    pinball_tot_lqr=0
    # compute pinball loss scores on test data
    for i, q in enumerate(quantiles):
        print(f"{q}: ", mean_pinball_loss(y_test,qr_models[i].predict(X_test), alpha=q)/np.mean(y_test))
        pinball_scores.loc[i,"Linear qr"]=mean_pinball_loss(y_test,y_test_pred_qr[i], alpha=q)/np.mean(y_test)
        pinball_tot_lqr+=mean_pinball_loss(y_test,y_test_pred_qr[i], alpha=q)/np.mean(y_test)*1/9
    
    pinball_scores.loc[9,"Linear qr"]=pinball_tot_lqr

    # define loss to tune
    neg_mean_pinball_loss_scorer_05 = make_scorer(
    mean_pinball_loss,
    alpha=0.5,
    greater_is_better=False,  # maximize the negative of the loss
    )
    
    # gbm quantile regressor
    # set grid parameters for hypertuning
    qr_gbr_models=[]
    y_test_pred_qr_gbr=[]

    param_grid_gbr = dict(
    learning_rate=[0.05, 0.1, 0.2],
    max_depth=[2, 5, 10],
    min_samples_leaf=[1, 5, 10, 20],
    min_samples_split=[5, 10, 20, 30, 50],
    )
    
    gbr_blueprint=gbr(loss="quantile", alpha=0.5, random_state=0)
    # tune hyperparameters
    # and fit data
    best_hyperparameters_gbm=HalvingRandomSearchCV(
            gbr_blueprint,
            param_grid_gbr,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer_05,
            n_jobs=2,
            random_state=0,
        ).fit(X_train, y_train).best_params_
    
    pinball_tot_gbm_qr=0
    for i,q in enumerate(tqdm(quantiles)):

        # fit data for specific quantile
        qr_gbr_models+=[gbr(loss="quantile", alpha=q, random_state=0,**best_hyperparameters_gbm).fit(X_train, y_train)]

        # list of prediction for each quantile
        y_test_pred_qr_gbr+=[qr_gbr_models[i].predict(X_test)]

        print(f"{q}: ",mean_pinball_loss(y_test,qr_gbr_models[i].predict(X_test), alpha=q)/np.mean(y_test))

        pinball_scores.loc[i,"Gbm qr"]=mean_pinball_loss(y_test,y_test_pred_qr_gbr[i], alpha=q)/np.mean(y_test)
        pinball_tot_gbm_qr+=mean_pinball_loss(y_test,y_test_pred_qr_gbr[i], alpha=q)/np.mean(y_test)*1/9
    
    pinball_scores.loc[9,"Gbm qr"]=pinball_tot_gbm_qr

    # save
    [pickle.dump(qr_gbr_models[i], open(f"/Users/luca/Desktop/kernel_quantile_regression/train_test/{country}/models_gbm_qr_{exp}/gbm_qr_{i}", "wb")) for i in range(9)]


    # ranform forest quantile regression
    qr_rfr_models=[]
    y_test_pred_qr_rfr=[]

    param_grid_rfr = dict(
    max_depth=[2, 5, 10],
    min_samples_leaf=[1, 5, 10, 20],
    min_samples_split=[5, 10, 20, 30, 50],
    )
    rfr_blueprint=rfr(default_quantiles=0.5)
    best_hyperparameters_rff=HalvingRandomSearchCV(
            rfr_blueprint,
            param_grid_rfr,
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer_05,
            n_jobs=2,
            random_state=0,
        ).fit(X_train, y_train).best_params_
    
    pinball_tot_qf_rfr=0
    for i,q in enumerate(tqdm(quantiles)):

        # fit data for specific quantile
        qr_rfr_models+=[rfr(default_quantiles=q, **best_hyperparameters_rff).fit(X_train, y_train)]

        # list of prediction for each quantile
        y_test_pred_qr_rfr+=[qr_rfr_models[i].predict(X_test)]
      
        print(f"{q}: ", mean_pinball_loss(y_test,qr_rfr_models[i].predict(X_test), alpha=q)/np.mean(y_test))

        pinball_scores.loc[i,"Quantile forest"]=mean_pinball_loss(y_test,y_test_pred_qr_rfr[i], alpha=q)/np.mean(y_test)
        pinball_tot_qf_rfr+=mean_pinball_loss(y_test,y_test_pred_qr_rfr[i], alpha=q)/np.mean(y_test)*1/9
    
    pinball_scores.loc[9,"Quantile forest"]=pinball_tot_qf_rfr

        
    # save
    [pickle.dump(qr_rfr_models[i], open(f"/Users/luca/Desktop/kernel_quantile_regression/train_test/{country}/models_qf_{exp}/qf_{i}", "wb")) for i in range(9)]

    # save scores
    pinball_scores.to_csv(f"train_test/{country}/scores_{exp}.csv", index=False)
# CH

# 0.1:  0.042578561510814496
# 0.2:  0.07460118502033142
# 0.3:  0.10011636632456154
# 0.4:  0.1190962110110857
# 0.5:  0.13095159624360198
# 0.6:  0.13425750284506457
# 0.7:  0.12855423773279462
# 0.8:  0.11201134824429139
# 0.9:  0.07884352843506973


# GBM 
# 0.1:  0.019086831563263656
# 2:  0.03075895568438049
# 3:  0.038376501334729106
# 4:  0.04310644741565326
# 5:  0.04498542411877865
# 6:  0.0438668505752885
# 7:  0.039280318182637816
# 8:  0.03082919813323486
# 9:  0.018781257262539017


# QF 
# 0.1 0.018799057672365196
# 0.2 0.030696640559951867
# 0.3 0.038672643359191204
# 0.4 0.04345989146587304
# 0.5 0.044903824661470755
# 0.6 0.04362922365768885
# 0.7 0.03916365178854014
# 0.8 0.031031521618896517
# 0.9 0.01870107869321539


# 0.1  &  0.018955529327865296  \\ 
# 0.2  &  0.03041218559045541  \\ 
# 0.3  &  0.038140623749173874  \\ 
# 0.4  &  0.042941513539870114  \\ 
# 0.5  &  0.04465625762074871  \\ 
# 0.6  &  0.04347250511620204  \\ 
# 0.7  &  0.03894392574179358  \\ 
# 0.8  &  0.030662643640494907  \\ 
# 0.9  &  0.018412788712590618  \\ 





# DE
# Linear
# 0.1:  0.04516866951010013
# 0.2:  0.08086415307130926
# 0.3:  0.10820128980640165
# 0.4:  0.12721878998203479
# 0.5:  0.1388625272101438
# 0.6:  0.1418185498659333
# 0.7:  0.13590791983193012
# 0.8:  0.11869441875427791
# 0.9:  0.08458144144296441


# GBM
# 0.1:  0.023815800357731932
# 2:  0.04130666491351428
# 3:  0.053394973088519794
# 4:  0.06128472541818242
# 5:  0.06498273930989444
# 6:  0.06282424525932778
# 7:  0.05498423782997457
# 8:  0.04120346408337301
# 9:  0.023039934701239367




# QF 
# 0.1 0.028900103953426754
# 0.2 0.047446415911667636
# 0.3 0.06130275856650811
# 0.4 0.07030793572026525
# 0.5 0.0737134407136041
# 0.6 0.071293744252726
# 0.7 0.06226763992674598
# 0.8 0.04740540958900103
# 0.9 0.02687381604411017

# 0.1  &  0.02766768075186939  \\ 
# 0.2  &  0.04669594000314759  \\ 
# 0.3  &  0.059813168136330204  \\ 
# 0.4  &  0.06833708708610238  \\ 
# 0.5  &  0.0724663410454275  \\ 
# 0.6  &  0.07053671769584387  \\ 
# 0.7  &  0.06215327171738864  \\ 
# 0.8  &  0.047298247827320124  \\ 
# 0.9  &  0.026978592995359534  \\ 