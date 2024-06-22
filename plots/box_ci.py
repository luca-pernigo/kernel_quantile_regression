# code for box plot in scatter plot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df=pd.read_csv("Data/Load/Task 9/L9-model_prediction_laplacian.csv")
df_test=pd.read_csv(f"Data/Load/Task 9/L9-test_clean.csv")
# get subset of interesting columns
df=df.iloc[:, np.r_[0, 2:101]]

# how many data to plot
n=48



df_res=pd.DataFrame(columns=["TIMESTAMP", "LOAD"])

for i in range(n):
    temp=df.iloc[i,:].values
    # timestamp
    timestamp=[i]*99
    val=temp[1:]

    # create sub df
    data={"TIMESTAMP":timestamp, "LOAD":val}
    df_sub=pd.DataFrame(data)

    df_res=pd.concat([df_res, df_sub], ignore_index=True)


# boxplot
fig, ax = plt.subplots(figsize=(15,5))
df_res.boxplot(column=["LOAD"], by="TIMESTAMP", showfliers=False, ax=ax)

plt.suptitle("")
plt.title("Boxplot task 9, laplacian kernel")

plt.ylabel("Load (MW)")
plt.xlabel("Hours")

# ticks whole data
# ticks for the days, (including also hours adds clutter)
# tick_array = [i for i in range(24,n+1,24)]
# # label ticks
# labels=[int(i/24) for i in tick_array]
# plt.xticks(ticks=tick_array, labels=labels, rotation=45)
# print(labels)


plt.show()