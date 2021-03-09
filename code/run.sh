# #! /usr/bin/env bash
# ## 1 & 3 Pruning 



python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --prune_val=0.1
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --prune_val=0.2
python3 main/main.py --lambda_loss=0.75  --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --prune_val=0.3
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --prune_val=0.4
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --prune_val=0.5
python3 main/main.py --lambda_loss=0.75  --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --prune_val=0.6
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --prune_val=0.7
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --prune_val=0.8
python3 main/main.py --lambda_loss=0.75  --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --prune_val=0.9


# python3 main/main.py --lambda_loss=0.50 --batch_size=16 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31
# python3 main/main.py --lambda_loss=0.75 --batch_size=16 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31
# python3 main/main.py --lambda_loss=1  --batch_size=16 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31


python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31 --prune_val=0.1
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31 --prune_val=0.2
python3 main/main.py --lambda_loss=0.75  --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31 --prune_val=0.3
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31 --prune_val=0.4
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31 --prune_val=0.5
python3 main/main.py --lambda_loss=0.75  --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31 --prune_val=0.6
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31 --prune_val=0.7
python3 main/main.py --lambda_loss=0.75 --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31 --prune_val=0.8
python3 main/main.py --lambda_loss=0.75  --batch_size=8 --architecture="Xception" --resize=299  --epochs=40 --combination="Webcam_to_Amazon" --output_classes=31 --prune_val=0.9
