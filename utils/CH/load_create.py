# script to clean load data from energy charts CH
import pandas as pd

import os
import sys

wd = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(wd, '..', '..'))
sys.path.append(project_directory)
from utils.miscellaneous import en_clean_load

df=pd.read_csv("Data/CH/CH_load.csv")

df=en_clean_load(df)
df.reset_index(inplace=True, drop=True)

df.to_csv("Data/CH/clean/load.csv", index=False)