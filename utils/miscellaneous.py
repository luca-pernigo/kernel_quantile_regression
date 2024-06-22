
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import numpy as np

import pandas as pd
import re

import holidays

def clean_time(file, date_start, date_end, start_hour):
    # read file as pandas df
    df=pd.read_csv(file, sep=",", decimal=".")
    
    ## clean data dates
    dates = pd.date_range(start=f'{date_start}-01-01 0{start_hour}:00:00', periods=len(df), freq='H')
    df['DAY'] = dates.day
    df['MONTH'] = dates.month
    df['HOUR'] = dates.hour
    df['YEAR'] = dates.year
    
    # add info weekday; saturday, sunday
    df['DAY_OF_WEEK'] = dates.dayofweek

    # holidays
    hol=holidays.US(years=range(date_start, date_end))
    df["IS_HOLIDAY"]=pd.Series(dates.date).isin(hol).astype(int)  
    # print(df)
    return df

def order_columns(df, col_list):
    df=df[col_list]
    return df

def get_task_number(file):
    pattern = r'(\d+)'

    matches = re.findall(pattern, file)

    if matches:
        n = int(matches[0])
        
    return n



def get_test(file):
    pattern = r'(\d+)'

    matches = re.findall(pattern, file)

    if matches:
        n = int(matches[0])
        n_new = n + 1

        file_new = re.sub(pattern, str(n_new), file)
    
    return file_new



def order_quantiles(df):
    # get 99 quantiles data only
    a = df[[f"{i/100}" for i in range(1,100)]].values
    # get additional columns that do not have to be sorted
    add_cols=df.columns.difference([f"{i/100}" for i in range(1,100)])
    
    a.sort(axis=1)
    # create df
    res=pd.DataFrame(a, df.index, columns=[f"{i/100}" for i in range(1,100)])

    res[add_cols]=df[add_cols]
    # reorder cols
    res=order_columns(res, add_cols.values.tolist()+[f"{i/100}" for i in range(1,100)])
    
    return res



def datetime_conv(df, time_col):
    df['TIMESTAMP'] = pd.to_datetime(df[f'{time_col}'], format='%m%d%Y %H:%M')

    df['DAY'] = df['TIMESTAMP'].dt.day
    df['MONTH'] = df['TIMESTAMP'].dt.month
    df['HOUR'] = df['TIMESTAMP'].dt.hour
    df['YEAR'] = df['TIMESTAMP'].dt.year

    # add info weekday; saturday, sunday
    df['DAY_OF_WEEK'] = df['TIMESTAMP'].dt.dayofweek

    # holidays
    hol=holidays.US(years=range(2001, 2011))
    df["IS_HOLIDAY"]=df['TIMESTAMP'].isin(hol)
    # convert it to categorical variable
    df["IS_HOLIDAY"]=pd.Categorical(df["IS_HOLIDAY"].astype(int))

    return df



# function for etl and cleaning data from energy charts
def clean_temperatures(df):
    
    # first row is addtional text clutter
    df=df.iloc[1:,:].copy()
    # rename col
    df.rename(columns={df.columns[0]:"Time", df.columns[1]:"Temperature"}, inplace=True)
    return df

def clean_winds(df):
    
    # first row is addtional text clutter
    df=df.iloc[1:,:].copy()
    # rename col
    df.rename(columns={df.columns[0]:"Time", df.columns[1]:"Wind_speed"}, inplace=True)
    return df


# function for etl and cleaning data from energy charts
def en_clean_load(df):
    # first row is addtional text clutter
    df=df.iloc[1:,:].copy()
    # rename col
    df.rename(columns={df.columns[0]:"Time", df.columns[1]:"Load"}, inplace=True)
    return df



def qs_plot(df, figsize, y_test):
    # Set1 colormap
    colors = cm.Set1([0,1,2,3,4,5,6,7,8])
    plt.figure(figsize=figsize)
    stack_plot(colors, df, y_test)
    
    set_legend()

    

def stack_plot(colors, df, y_test):
    cum = 0
    for i, color in enumerate(colors):
        cum += df.iloc[:,i]
        
        plt.plot(cum,color=colors[i] , label=f"q = {10*(i+1)}%", linewidth = 0.5)

        if (i==3):
            # print("aaa")
            off=cum.copy()

    plt.plot(off+y_test, color="black", linewidth=1, label="effective")


def set_legend():
    # function to set legend outside of plot vertical horientation
    leg = plt.legend(bbox_to_anchor=(1, 1))
    


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
    plt.plot(y,color="black", label="effective")
    
    plt.fill_between(x,y_predict_5,y, alpha=0.4, color="green", edgecolor="red")
    plt.fill_between(x,y,y_predict_95, alpha=0.4, color="green", edgecolor="red", label="90% Confidence interval")
    
    plt.xticks(ticks=tick_array, labels=labels, rotation=45)
    plt.xlabel("Days")
    plt.ylabel("Load (MW)")

    plt.legend()
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
    plt.plot(y,color="black", label="effective")
    
    plt.fill_between(x,y_predict_5,y, alpha=0.4, color="green", edgecolor="red")
    plt.fill_between(x,y,y_predict_95, alpha=0.4, color="green", edgecolor="red", label="90% Confidence interval")
    
    plt.xticks(ticks=tick_array, rotation=45)
    plt.xlabel("Hours")
    plt.ylabel("Price ($/MW)")

    plt.legend()
    return None
