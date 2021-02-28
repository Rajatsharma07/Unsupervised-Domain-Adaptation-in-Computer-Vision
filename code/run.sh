# #! /usr/bin/env bash
# ## 1st

# python3 main/main.py --prune_val=0.20 --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam"
# python3 main/main.py --prune_val=0.30 --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam"
# python3 main/main.py --prune_val=0.40 --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam"
# python3 main/main.py --prune_val=0.50 --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam"
# python3 main/main.py --prune_val=0.60 --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam"
# python3 main/main.py --prune_val=0.70 --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam"
# python3 main/main.py --prune_val=0.80 --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam"
# python3 main/main.py --prune_val=0.90 --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam"
# python3 main/main.py --prune_val=1 --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam"


#python3 main/main.py --lambda_loss=0.50 --batch_size=8 --architecture="Xception" --resize=71  --epochs=50 --combination="SynSigns_to_GTSRB"
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=71  --epochs=40 --combination="SynSigns_to_GTSRB"
python3 main/main.py --lambda_loss=1 --batch_size=8 --architecture="Xception" --resize=71  --epochs=40 --combination="SynSigns_to_GTSRB"