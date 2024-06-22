# script to clean load data from energy charts CH
import pandas as pd

import os
import sys

from sec_met import clean_load

# script to clean the load data from energy charts
# CH
clean_load("energy-charts_Public_net_electricity_generation_in_Switzerland_in_2024", "CH", "test", 2024)
clean_load("energy-charts_Public_net_electricity_generation_in_Switzerland_in_2023", "CH", "test", 2023)
clean_load("energy-charts_Public_net_electricity_generation_in_Switzerland_in_2022", "CH", "test", 2022)
# clean_load("energy-charts_Public_net_electricity_generation_in_Switzerland_in_2021", "CH", "test")
# clean_load("energy-charts_Public_net_electricity_generation_in_Switzerland_in_2020", "CH", "train")


# DE
clean_load("energy-charts_Public_net_electricity_generation_in_Germany_in_2024", "DE", "test", 2024)
clean_load("energy-charts_Public_net_electricity_generation_in_Germany_in_2023", "DE", "test", 2023)
clean_load("energy-charts_Public_net_electricity_generation_in_Germany_in_2022", "DE", "test", 2022)
# clean_load("energy-charts_Public_net_electricity_generation_in_Germany_in_2021", "DE", "test")
# clean_load("energy-charts_Public_net_electricity_generation_in_Germany_in_2020", "DE", "train")

# AT
clean_load("energy-charts_Public_net_electricity_generation_in_Austria_in_2024", "AT", "test", 2024)
clean_load("energy-charts_Public_net_electricity_generation_in_Austria_in_2023", "AT", "test", 2023)
clean_load("energy-charts_Public_net_electricity_generation_in_Austria_in_2022", "AT", "test", 2022)
# clean_load("energy-charts_Public_net_electricity_generation_in_Austria_in_2021", "AT", "test")
# clean_load("energy-charts_Public_net_electricity_generation_in_Austria_in_2020", "AT", "train")




