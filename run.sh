#! /usr/bin/env bash

python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture=3 --combination=2 --resize=299
python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=2 --resize=299
python3 main/main.py --lambda_loss=1.0 --batch_size=8  --architecture=3 --combination=2 --resize=299