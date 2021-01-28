#! /usr/bin/env bash

python3 main/main.py --lambda_loss=0.75 --batch_size=8 --freeze_upto=7 --architecture=3 --combination=2 --resize=299 --epochs=25
# python3 main/main.py --lambda_loss=0.50 --batch_size=8 --freeze_upto=7 --architecture=3 --combination=2 --resize=299
# python3 main/main.py --lambda_loss=1.0 --batch_size=8 --freeze_upto=7 --architecture=3 --combination=2 --resize=299