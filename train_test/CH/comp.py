# competitor performance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pprint import pprint

from quantile_forest import RandomForestQuantileRegressor as rfr

from sklearn.ensemble import GradientBoostingRegressor as gbr

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, GridSearchCV
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
    country="CH"
    exp="full"
    
    # quantiles
    quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    # load data
    train=pd.read_csv(f"Data/{country}/2021/clean/{country.lower()}.csv")
    test=pd.read_csv(f"Data/{country}/2022/clean/{country.lower()}.csv")
    pinball_scores=pd.read_csv(f"train_test/{country}/scores_{exp}.csv")
    
    # X y
    X_train=train[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]]
    X_test=test[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]]

    y_train=train["Load"]
    y_test=test["Load"]

   
    # linear quantile regression
    qr_models = [qr.QuantReg(y_train, X_train).fit(q=q) for q in quantiles]
    
    y_test_pred_qr=[qr_model.predict(X_test) for qr_model in qr_models]
    

    # save 
    [pickle.dump(qr_models[i], open(f"train_test/{country}/models_lqr_{exp}/lqr_{i}.pkl", "wb")) for i in range(9)]
    
    
    pinball_tot_lqr=0
    # compute pinball loss scores on test data
    for i, q in enumerate(quantiles):
        print(f"{q}: ", mean_pinball_loss(y_test,qr_models[i].predict(X_test), alpha=q)/np.mean(y_test))
        pinball_scores.loc[i,"Linear qr"]=mean_pinball_loss(y_test,y_test_pred_qr[i], alpha=q)/np.mean(y_test)
        pinball_tot_lqr+=mean_pinball_loss(y_test,y_test_pred_qr[i], alpha=q)/np.mean(y_test)*1/9
    
    print("CRPS: ", pinball_tot_lqr)
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
    for i,q in enumerate(quantiles):

        # fit data for specific quantile
        qr_gbr_models+=[gbr(loss="quantile", alpha=q, random_state=0,**best_hyperparameters_gbm).fit(X_train, y_train)]

        # list of prediction for each quantile
        y_test_pred_qr_gbr+=[qr_gbr_models[i].predict(X_test)]

        print(f"{q}: ",mean_pinball_loss(y_test,qr_gbr_models[i].predict(X_test), alpha=q)/np.mean(y_test))

        pinball_scores.loc[i,"Gbm qr"]=mean_pinball_loss(y_test,y_test_pred_qr_gbr[i], alpha=q)/np.mean(y_test)
        pinball_tot_gbm_qr+=mean_pinball_loss(y_test,y_test_pred_qr_gbr[i], alpha=q)/np.mean(y_test)*1/9
    
    print("CRPS: ", pinball_tot_gbm_qr)
    pinball_scores.loc[9,"Gbm qr"]=pinball_tot_gbm_qr

    # save
    [pickle.dump(qr_gbr_models[i], open(f"train_test/{country}/models_gbm_qr_{exp}/gbm_qr_{i}.pkl", "wb")) for i in range(9)]


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
            random_state=10,
        ).fit(X_train.values, y_train.values).best_params_
    
    pinball_tot_qf_rfr=0
    for i,q in enumerate(quantiles):

        # fit data for specific quantile
        qr_rfr_models+=[rfr(default_quantiles=q, **best_hyperparameters_rff).fit(X_train.values, y_train.values)]

        # list of prediction for each quantile
        y_test_pred_qr_rfr+=[qr_rfr_models[i].predict(X_test.values)]
      
        print(f"{q}: ", mean_pinball_loss(y_test,qr_rfr_models[i].predict(X_test.values), alpha=q)/np.mean(y_test))

        pinball_scores.loc[i,"Quantile forest"]=mean_pinball_loss(y_test,y_test_pred_qr_rfr[i], alpha=q)/np.mean(y_test)
        pinball_tot_qf_rfr+=mean_pinball_loss(y_test,y_test_pred_qr_rfr[i], alpha=q)/np.mean(y_test)*1/9
    
    print("CRPS: ", pinball_tot_qf_rfr)
    pinball_scores.loc[9,"Quantile forest"]=pinball_tot_qf_rfr

        
    # save
    [pickle.dump(qr_rfr_models[i], open(f"train_test/{country}/models_qf_{exp}/qf_{i}.pkl", "wb")) for i in range(9)]

    # save scores
    pinball_scores.to_csv(f"train_test/{country}/scores_{exp}.csv", index=False)

# CH

# 0.1:  0.035946942609544716
# 0.2:  0.0616104688625966
# 0.3:  0.08063685598384654
# 0.4:  0.09395974649071308
# 0.5:  0.10223523331817143
# 0.6:  0.10392969811048319
# 0.7:  0.09892291027181377
# 0.8:  0.0852824944425314
# 0.9:  0.058919221429465415
# CRPS:  0.0801603968354629


# GBM 
# 0.1:  0.01242811307607976
# 0.2:  0.019936230570722028
# 0.3:  0.02573120053493338
# 0.4:  0.02949830323213118
# 0.5:  0.03174380374863501
# 0.6:  0.031811499598426725
# 0.7:  0.03008910007582111
# 0.8:  0.025701288729266265
# 0.9:  0.01862263519899499
# CRPS:  0.02506246386277894


# QF 
# 0.1:  0.014568108959131984
# 0.2:  0.022766514209163508
# 0.3:  0.02744529323025672
# 0.4:  0.030379021415090947
# 0.5:  0.03118400769403946
# 0.6:  0.030324031731645416
# 0.7:  0.02758657080366495
# 0.8:  0.023524118911529068
# 0.9:  0.015115323962960807
# CRPS:  0.024765887879720318

# 0.1  &  0.012097906860201567  \\ 
# 0.2  &  0.01961847445279765  \\ 
# 0.3  &  0.024945762466853184  \\ 
# 0.4  &  0.028533396674224894  \\ 
# 0.5  &  0.030478723916750956  \\ 
# 0.6  &  0.031090658437225555  \\ 
# 0.7  &  0.02958217123128587  \\ 
# 0.8  &  0.025805227217416106  \\ 
# 0.9  &  0.018450239815096255  \\ 
# total quantile loss:  0.024511395674650226



# DE
# Linear
# 0.1:  0.04050873481892131
# 0.2:  0.0695875235494056
# 0.3:  0.09068652634480819
# 0.4:  0.10542751705128771
# 0.5:  0.11379281126089467
# 0.6:  0.11543307992065718
# 0.7:  0.10933982661308293
# 0.8:  0.0931691012854489
# 0.9:  0.06318525856272718
# CRPS:  0.08901448660080373


# GBM
# 0.1:  0.026783235550629874
# 0.2:  0.030202574174767818
# 0.3:  0.031100926143492604
# 0.4:  0.030110853067439474
# 0.5:  0.027819850911379018
# 0.6:  0.02485400719167576
# 0.7:  0.02116092868960098
# 0.8:  0.016504598536059664
# 0.9:  0.010496230347500291
# CRPS:  0.024337022734727275




# QF 
# 0.1:  0.016554259985186667
# 0.2:  0.024168612397068443
# 0.3:  0.028546426824771064
# 0.4:  0.030579090503860738
# 0.5:  0.030875777269190618
# 0.6:  0.029676940516877123
# 0.7:  0.02612141958076377
# 0.8:  0.020986964196431056
# 0.9:  0.012888690616181517
# CRPS:  0.024488686876703445

# 0.1  &  0.017736456269952385  \\ 
# 0.2  &  0.025165632302538318  \\ 
# 0.3  &  0.02794766181932907  \\ 
# 0.4  &  0.02881216273211498  \\ 
# 0.5  &  0.02786590763228558  \\ 
# 0.6  &  0.02557669030729824  \\ 
# 0.7  &  0.022080120632472982  \\ 
# 0.8  &  0.017367587497682996  \\ 
# 0.9  &  0.01127868757443536  \\ 
# total quantile loss:  0.02264787852978999