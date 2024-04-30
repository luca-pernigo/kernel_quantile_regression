from datetime import datetime
import pandas as pd

import os
import sys

wd = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(wd, '..', '..'))
sys.path.append(project_directory)
from utils.miscellaneous import clean_temperatures


year=2022

df=pd.read_csv(f"Data/CH/{year}/wind_speed.csv")

df.rename(columns={df.columns[0]:"Time", df.columns[1]:"Wind_speed"}, inplace=True)

df.to_csv(f"Data/CH/{year}/clean/wind_speed.csv", index=False)