import matplotlib.pyplot as plt
import pandas as pd

import os
import sys



from sec_met import clean_sec_met_test


# script to clean SECURES-Met data into test data
# CH

clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/T2M_NUTS0_Europe_popweight_rcp45_hourly_2001-2050.csv", "CH","Temperature", "temperature")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/BNI_NUTS0_Europe_popweight_rcp45_hourly_2001-2050.csv", "CH","Direct_irradiation", "direct_irradiation")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/GLO_NUTS0_Europe_popweight_rcp45_hourly_2001-2050.csv", "CH","Global_radiation", "global_radiation")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/WP_NUTS0_Europe_potential-area_rcp45_hourly_1981-2100.csv", "CH","Wind_potential", "wind_potential")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/HYD-RES_NUTS0_Europe_sum_rcp45_daily_2006-2100.csv", "CH","Hydro_reservoir", "hydro_reservoir")

clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/HYD-ROR_NUTS0_Europe_sum_rcp45_daily_2006-2100.csv", "CH","Hydro_river", "hydro_river")


# DE
clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/T2M_NUTS0_Europe_popweight_rcp45_hourly_2001-2050.csv", "DE","Temperature", "temperature")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/BNI_NUTS0_Europe_popweight_rcp45_hourly_2001-2050.csv", "DE","Direct_irradiation", "direct_irradiation")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/GLO_NUTS0_Europe_popweight_rcp45_hourly_2001-2050.csv", "DE","Global_radiation", "global_radiation")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/WP_NUTS0_Europe_potential-area_rcp45_hourly_1981-2100.csv", "DE","Wind_potential", "wind_potential")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/HYD-RES_NUTS0_Europe_sum_rcp45_daily_2006-2100.csv", "DE","Hydro_reservoir", "hydro_reservoir")

clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/HYD-ROR_NUTS0_Europe_sum_rcp45_daily_2006-2100.csv", "DE","Hydro_river", "hydro_river")

# AT

clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/T2M_NUTS0_Europe_popweight_rcp45_hourly_2001-2050.csv", "AT","Temperature", "temperature")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/BNI_NUTS0_Europe_popweight_rcp45_hourly_2001-2050.csv", "AT","Direct_irradiation", "direct_irradiation")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/GLO_NUTS0_Europe_popweight_rcp45_hourly_2001-2050.csv", "AT","Global_radiation", "global_radiation")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/WP_NUTS0_Europe_potential-area_rcp45_hourly_1981-2100.csv", "AT","Wind_potential", "wind_potential")


clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/HYD-RES_NUTS0_Europe_sum_rcp45_daily_2006-2100.csv", "AT","Hydro_reservoir", "hydro_reservoir")

clean_sec_met_test("/Users/luca/Downloads/Future_RCP45/NUTS0_Europe/HYD-ROR_NUTS0_Europe_sum_rcp45_daily_2006-2100.csv", "AT","Hydro_river", "hydro_river")