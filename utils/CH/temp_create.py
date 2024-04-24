from datetime import datetime
import pandas as pd

import os
import sys

wd = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(wd, '..', '..'))
sys.path.append(project_directory)
from utils.miscellaneous import clean_temperatures

df=pd.read_csv("Data/CH/temperature.csv")

df.rename(columns={df.columns[0]:"Time", df.columns[1]:"Temperature"}, inplace=True)

df.to_csv("Data/CH/clean/temperature.csv", index=False)