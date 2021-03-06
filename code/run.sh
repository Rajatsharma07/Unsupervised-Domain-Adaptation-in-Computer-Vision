# #! /usr/bin/env bash
# ## 1st

python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=71  --epochs=40 --combination="SynSigns_to_GTSRB"

python3 main/main.py --lambda_loss=1  --batch_size=8 --architecture="Xception" --resize=71  --epochs=40 --combination="SynSigns_to_GTSRB"


python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=71  --epochs=40 --combination="SynSigns_to_GTSRB"