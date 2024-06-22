# code for box plot in scatter plot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pickle
from sklearn.preprocessing import StandardScaler
import sys


def box_ci(data_path,u,l,n,title,subset=False,test_path=None):
    df=pd.read_csv(data_path)
    if test_path!=None:
        df_test=pd.read_csv(test_path)
    if subset==True:
        # get subset of interesting columns
        df=df.iloc[:, np.r_[2:101]]

    if n==-1:
        n=len(df)

    print(df)

    # init df_res, create columns for timestamp and load
    df_res=pd.DataFrame(columns=["TIMESTAMP", "LOAD"])

    for i in range(l,u):
        temp=df.iloc[i,:].values
        # timestamp
        timestamp=[i]*(len(df.columns))
        val=temp[0:]

        # create sub df
        data={"TIMESTAMP":timestamp, "LOAD":val}
        df_sub=pd.DataFrame(data)

        df_res=pd.concat([df_res, df_sub], ignore_index=True)

    print(df_res)
    # boxplot
    fig, ax = plt.subplots(figsize=(15,5))
    df_res.boxplot(column=["LOAD"], by="TIMESTAMP", showfliers=False, ax=ax)

    plt.suptitle("")
    plt.title(f"{title}")

    plt.ylabel("Load (MW)")
    plt.xlabel("Hours")

    # ticks whole data
    # ticks for the days, (including also hours adds clutter)
    # tick_array = [i for i in range(24,n+1,24)]
    # # label ticks
    # labels=[int(i/24) for i in tick_array]
    plt.xticks(ticks=np.linspace(1,u-l+1,u-l), labels=[i for i in range(u-l)], rotation=45)
    # print(labels)

    if test_path!=None:
        try:
            plt.plot(np.linspace(1,u-l+1,u-l), df_test["Load"][l:u].values)
        except:
            plt.plot(np.linspace(1,u-l+1,u-l), df_test["LOAD"][l:u].values)

    plt.show()


if __name__=="__main__":
    box_ci(data_path="Data/Load/Task 9/L9-model_prediction_laplacian.csv", l=0,u=48,n=48,title="Boxplot task 9, Absolute Laplacian kernel",subset=True)

    # select country
    country="CH"
    country_name="Switzerland"
    ktype="a_laplacian"

    box_ci(data_path=f"Data/{country}/2022/clean/model_prediction_{ktype}.csv", l=3300,u=3400,n=300,title="Boxplot probabilistic forecast for load in Switzerland (2022)",subset=False, test_path=f"Data/{country}/2022/clean/{country.lower()}.csv")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# df_pred=pd.read_csv(f"Data/{country}/2022/clean/model_prediction_{ktype}.csv")
# # train=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/{country}/2021/clean/{country.lower()}.csv")
# print(df_pred)
# # test=pd.read_csv(f"/Users/luca/Desktop/kernel_quantile_regression/Data/{country}/2022/clean/{country.lower()}.csv")

# # quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# # # X y
# # X_train=train[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]]
# # X_test=test[["Temperature","Wind_speed", "Hour","Day_of_week","Month","Is_holiday"]]

# # y_train=train["Load"]
# # y_test=test["Load"]

# # # scale data
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)



# # # confidence interval
# # # plot first predictions
# # n=100
# # plt.figure(figsize=(15,5))
# # x=np.linspace(0,len(y_test),num=len(y_test))[0:n]
# # plt.plot(x,y_test[0:n], color="black", label="effective")

# # sys.path.append("/Users/luca/Desktop/kernel_quantile_regression/")
# # # 95%
# # krn_q=pickle.load(open(f"/Users/luca/Desktop/kernel_quantile_regression/train_test/{country}/models/krn_qr_{0.95}.pkl", "rb"))
# # y_predict_95=krn_q.predict(X_test_scaled)
# # plt.fill_between(x,y_predict_95[0:n],y_test[0:n], alpha=0.4, color="green", edgecolor="red", label="90% Confidence interval")

# # # 5%
# # krn_q=pickle.load(open(f"train_test/{country}/models/krn_qr_{0.05}.pkl", "rb"))
# # y_predict_05=krn_q.predict(X_test_scaled)

# # plt.fill_between(x,y_test[0:n], y_predict_05[0:n], alpha=0.4, color="green", edgecolor="red")
# # plt.ylabel("Load (MW)")

# # plt.legend()
# # plt.xlabel("Observations")
# # plt.title(f"Probabilistic forecast for load in {country_name} (2022)")

# # # savefig
# # # plt.savefig(f"plots/{country}/{country}_load_CI.png")

# # plt.show()
