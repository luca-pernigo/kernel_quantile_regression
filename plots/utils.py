# magick *.png -transparent white *.pdf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ktype_title={"a_laplacian":"Absolute Laplacian", "gaussian_rbf":"Gaussian RBF"}


def plot_ci(y_test,y_predict_95,y_predict_05,df_pred,u,l,country_name,year):
    dates=pd.date_range(f'{year}-01-01', periods=len(df_pred), freq='H')
    dates=pd.Series(dates)
    
    plt.figure(figsize=(25,5))
    x=np.linspace(0,u-l,u-l)

    # position of first days of month
    x_position=dates[ (dates.dt.hour==0) & (dates.dt.day==1)].index.values

    u_pos = np.absolute(x_position[x_position<=u]-u)
    u_idx=u_pos.argmin()

    l_pos = np.absolute(x_position[x_position<=l]-l)
    l_idx=l_pos.argmin()

    months=["Genuary","February","March","April","May","June","July","August","September","October","November","December"]
    # print(l_pos,l_idx,months[:l_idx+1],x_position[:l_idx+1])
    # print(x_position[l_idx:u_idx+1]-l)

    x_position=x_position[l_idx:u_idx+1]-l
    months_position=months[l_idx:u_idx+1]

    # if first element is negative drop
    if x_position[0]<0:
        x_position=x_position[1:]
        months_position=months_position[1:]


    plt.plot(x,y_test[l:u], color="black")


    # fill
    plt.fill_between(x,y_predict_95[l:u], y_predict_05[l:u], alpha=0.4, color="green", edgecolor="black", interpolate=True)

    plt.fill_between(x,y_predict_95[l:u],y_test[l:u],where=y_predict_95[l:u]<y_test[l:u], alpha=0.4, color="red", edgecolor="black", interpolate=True)

    plt.fill_between(x,y_test[l:u], y_predict_05[l:u], where=y_predict_05[l:u]>=y_test[l:u], alpha=0.4, color="red", edgecolor="black",interpolate=True)

    handles, labels = plt.gca().get_legend_handles_labels()

    # create manual symbols for legend
    patch1 = mpatches.Patch(color='green', label='90% Confidence interval',alpha=0.4)   
    patch2 = mpatches.Patch(color='red', label='Prediction out of confidence interval',alpha=0.4)   
    line = Line2D([0],[0],label="effective", color='black')

    # add manual symbols to auto legend
    handles.extend([line, patch1,patch2])

    plt.legend(handles=handles)

    # plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.title(f"Probabilistic forecast for load in {country_name} (2022)")
    plt.xticks(x_position, months_position)

    # make plot margins tight enough
    plt.margins(x=0.02)

