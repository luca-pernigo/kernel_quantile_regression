import numpy as np

import pandas as pd

from sklearn.metrics import mean_pinball_loss
# script for storing scores CH DE

quantiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
country="DE"
exp="full"


df_pred=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/{country}/2022/clean/model_prediction_{exp}.csv")
test=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/{country}/2022/clean/{country.lower()}.csv")
y_test=test["Load"]


try:
    pinball_scores=pd.read_csv(f"train_test/{country}/scores_{exp}.csv")

except:
    pinball_scores=pd.DataFrame(columns=["Quantile","Linear qr", "Gbm qr", "Quantile forest", "Kernel qr"])
    pinball_scores["Quantile"]=quantiles+["CRPS"]
    pinball_tot=0
    for i,q in enumerate(quantiles):
        pinball_scores.loc[i,"Kernel qr"]=mean_pinball_loss(y_test,df_pred[f"{q}"], alpha=q)/np.mean(y_test)
        pinball_tot+=mean_pinball_loss(y_test,df_pred[f"{q}"], alpha=q)/np.mean(y_test)*1/9
    pinball_scores.loc[9,"Kernel qr"]=pinball_tot


# print latex
print(pinball_scores.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.5f}".format))


pinball_scores.to_csv(f"train_test/{country}/scores_{exp}.csv", index=False)