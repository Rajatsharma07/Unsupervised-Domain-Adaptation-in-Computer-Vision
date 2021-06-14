import tensorflow as tf
from pathlib import Path
from modules.loss import CORAL, coral_loss

initializer = tf.keras.initializers.he_normal()  # Layer initializations

BASE_DIR = Path("/root/Master-Thesis/code")  # Base path
SYNTHETIC_PATH = BASE_DIR / Path("data/synthetic_data/")
OFFICE_DS_PATH = BASE_DIR / Path("data/office/")
GTSRB_PATH = Path("/root/Master-Thesis/code/data/GTSRB/")
LOGS_DIR = BASE_DIR / Path("logs/")  # Logs path
MODEL_PATH = BASE_DIR / Path("model_data/")  # Model path
EVALUATION = BASE_DIR / Path("evaluation/")  # Evalaution plots path
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_COMBINATION = {
    "Amazon_to_Webcam": 1,
    "Amazon_to_DSLR": 2,
    "Webcam_to_Amazon": 3,
    "DSLR_to_Amazon": 4,
    "SynSigns_to_GTSRB": 5,
}

ARCHITECTURE = {"Xception": 1, "Other": 2}
LOSS = {"CORAL": CORAL, "Another": coral_loss}
