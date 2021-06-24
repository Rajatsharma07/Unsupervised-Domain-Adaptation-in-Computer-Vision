
# Trigger MBM without data augmentation for A->W Scenario
python3 main/main.py --lambda_loss=0.50 --batch_size=16 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --save_model --save_weights

# Trigger CDAN without data augmentation for A->W Scenario
python3 main/main.py --lambda_loss=0.50 --batch_size=16 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --save_model --save_weights --technique

# Trigger MBM with data augmentation for A->W Scenario
python3 main/main.py --lambda_loss=0.50  --batch_size=16 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --save_model --save_weights --augment

# Trigger Pruned MBM without data augmentation for A->W Scenario
python3 main/main.py --lambda_loss=0.50 --batch_size=16 --architecture="Xception" --resize=299  --epochs=40 --combination="Amazon_to_Webcam" --output_classes=31 --save_model --save_weights --prune --prune_val=0.1
