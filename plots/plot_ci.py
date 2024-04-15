
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# plot confidence interval 90%
def load_plot_ci(df, y):
    # take data
    y_predict_5=df["0.05"]
    y_predict_50=df["0.5"]
    y_predict_95=df["0.95"]

    # dummy x for plotting
    x=np.arange(len(y))
    # ticks for the days, (including also hours adds clutter)
    tick_array = [i for i in range(24,len(y)+1,24)]
    # label ticks
    labels=[int(i/24) for i in tick_array]
    
    plt.figure(figsize=(15,5))
    # plot quantile range
    plt.plot(y,color="black")
    
    plt.fill_between(x,y_predict_5,y, alpha=0.4, color="green", edgecolor="red")
    plt.fill_between(x,y,y_predict_95, alpha=0.4, color="green", edgecolor="red")
    
    plt.xticks(ticks=tick_array, labels=labels, rotation=45)
    plt.xlabel("Days")
    plt.ylabel("Load in MW")
    return None


def price_plot_ci(df, y):
    # take data
    y_predict_5=df["0.05"]
    y_predict_50=df["0.5"]
    y_predict_95=df["0.95"]

    # dummy x for plotting
    x=np.arange(len(y))
    # ticks for the days, (including also hours adds clutter)
    tick_array = [i for i in range(0,23+1)]
    
    # plot quantile range
    plt.plot(y,color="black")
    
    plt.fill_between(x,y_predict_5,y, alpha=0.4, color="green", edgecolor="red")
    plt.fill_between(x,y,y_predict_95, alpha=0.4, color="green", edgecolor="red")
    
    plt.xticks(ticks=tick_array, rotation=45)
    plt.xlabel("Hours")
    plt.ylabel("Price in MWh")
    return None
