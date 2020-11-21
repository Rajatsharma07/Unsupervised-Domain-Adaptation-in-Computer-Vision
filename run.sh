#! /usr/bin/env bash


combinations=(ds1 ds2 ds3) 
models=(s1 s2)
seeds=(s1 s2 s3)

for comb in ${combinations[@]}; do
    for model in ${models[@]}; do
        for seed in ${seeds[@]}; do
            # python3 main.py --combination "$comb" --source_model "$model" --sample_seed "$seed"
            echo "python3 main.py --combination $comb --source_model $model --sample_seed $seed"
        done
    done
done