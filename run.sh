#! /usr/bin/env bash

# python main.py
# --combination=1 --source_model=1 --target_model="target_model" --batch_size=32 --resize=32 --learning_rate=0.001 --mode="train_test" --lamb da_loss=0.5
# --epochs=50 --save_weights=True --save_model=True --use_multiGPU=False

# seeds=(400, 1000, 250)
# losses=(0.25, 0.50, 0.75, 1)


# for loss in ${losses[@]}; do
#     for index in $(seq 1 3); do
#         echo "python3 main/main.py --lambda_loss="$loss" --batch_size=64"
#         python3 main/main.py --lambda_loss=$loss --batch_size=64
#     done
# done


python3 main/main.py --lambda_loss=1 --batch_size=64
python3 main/main.py --lambda_loss=1 --batch_size=64
python3 main/main.py --lambda_loss=1 --batch_size=64