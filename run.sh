#! /usr/bin/env bash

python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=1 --resize=299 --prune=True --prune_val=0.10 --epochs=40
python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=1 --resize=299 --prune=True --prune_val=0.20 --epochs=40
python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=1 --resize=299 --prune=True --prune_val=0.30 --epochs=40
python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=1 --resize=299 --prune=True --prune_val=0.40 --epochs=40
python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=1 --resize=299 --prune=True --prune_val=0.50 --epochs=40
python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=1 --resize=299 --prune=True --prune_val=0.60 --epochs=40
python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=1 --resize=299 --prune=True --prune_val=0.70 --epochs=40
python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=1 --resize=299 --prune=True --prune_val=0.80 --epochs=40
python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture=3 --combination=1 --resize=299 --prune=True --prune_val=0.90 --epochs=40