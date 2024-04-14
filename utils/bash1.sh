#!/bin/bash
# code for moving test data in respective folder
for((i=2;i<=15;i++))
do
    i_minus_1=$((i - 1))
    i_minus_1=$((i-1))
    # get test data
    cp "Data/Load/Task ${i}/L${i}-train.csv" "Data/Load/Task ${i_minus_1}/L${i_minus_1}-test.csv"
done

cp "Data/Load/Solution to Task 15/solution15_L_temperature.csv" "Data/Load/Task 15/L15-test.csv"
