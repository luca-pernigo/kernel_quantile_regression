import pandas as pd

import os
import sys

wd = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(wd, '..', '..'))
sys.path.append(project_directory)
from utils.miscellaneous import clean_temperatures

year=2022

# script to merge, clean and prepare the temperature dataset from energy charts
months=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# start empty df
df=pd.DataFrame(columns=["Time", "Temperature"])

for i,m in enumerate(months):
    # read clean append
    df_temp=pd.read_csv(f"Data/DE/{year}/Air_temperature/{m}.csv")
    df_temp=clean_temperatures(df_temp)

    df=pd.concat([df, df_temp], axis=0)

# rest index
df.reset_index(inplace=True, drop=True)

df.to_csv(f"Data/DE/{year}/clean/temperature.csv", index=False)
