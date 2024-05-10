import numpy as np

import pandas as pd

from sklearn.metrics import mean_pinball_loss
# script for storing scores SECURES-Met data

quantiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
country="AT"

test=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/SECURES-Met/{country}/clean/test/df.csv")
y_test=test["Load"]

kernels=["gaussian_rbf", "laplacian", "matern_1.5", "matern_2.5", "linear", "periodic", "polynomial", "sigmoid", "cosine"]

try:
    pinball_scores=pd.read_csv(f"train_test/SECURES-Met/{country}/scores.csv")

except:
    pinball_scores=pd.DataFrame(columns=["Quantile"]+kernels)
    pinball_scores["Quantile"]=quantiles+["CRPS"]

    for ktype in kernels:
        pinball_tot=0

        df_pred=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/SECURES-Met/{country}/clean/model_prediction_{ktype}.csv")

        for i,q in enumerate(quantiles):
            pinball_scores.loc[i,f"{ktype}"]=mean_pinball_loss(y_test,df_pred[f"{q}"], alpha=q)/np.mean(y_test)
            pinball_tot+=mean_pinball_loss(y_test,df_pred[f"{q}"], alpha=q)/np.mean(y_test)*1/9
        
        pinball_scores.loc[9,f"{ktype}"]=pinball_tot



pinball_scores.columns = ["Quantile"]+[k.capitalize().replace("_", " ") for k in kernels]

# print latex
print(pinball_scores.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.5f}".format))


pinball_scores.to_csv(f"train_test/SECURES-Met/{country}/scores.csv", index=False)