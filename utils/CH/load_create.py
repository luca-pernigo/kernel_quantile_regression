# script to clean load data from energy charts CH
import pandas as pd

import os
import sys

wd = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(wd, '..', '..'))
sys.path.append(project_directory)
from utils.miscellaneous import en_clean_load

year=2022

df=pd.read_csv(f"Data/CH/{year}/CH_load.csv")

df=en_clean_load(df)
df.reset_index(inplace=True, drop=True)

df.to_csv(f"Data/CH/{year}/clean/load.csv", index=False)