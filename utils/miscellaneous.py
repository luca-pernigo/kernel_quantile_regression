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
    a = df[[f"{i/100}" for i in range(1,100)]].values
    a.sort(axis=1)
    res=pd.DataFrame(a, df.index)
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
