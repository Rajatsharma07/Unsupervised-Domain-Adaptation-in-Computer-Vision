#! /usr/bin/env bash


combinations=(ds1 ds2 ds3) 
models=(s1 s2)
seeds=(s1 s2 s3)

for comb in ${combinations[@]}; do
    for model in ${models[@]}; do
        for seed in ${seeds[@]}; do
            # python3 main.py --combination "$comb" --source_model "$model" --sample_seed "$seed"
            echo "python3 main.py --combination $comb --source_model $model --sample_seed $seed --method --lsave"
        done
    done
    email "results for ds1" result-ds1
done

email "phase done"

for comb in ${combinations[]}; do
    for dir in "$comb*"; do
        results-ds1.csv
        sort -k2,2 -k2,1 readline "$dir/csv.csv"
         done
         done

email "results generated " <<< files
results.csv

test1=(1 2 3 4 5)

email "model save"
while val <<< readline  results.csv
    for tests in ${tests[@]}; done
