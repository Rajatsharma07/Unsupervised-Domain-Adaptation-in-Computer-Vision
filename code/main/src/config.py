import tensorflow as tf
from pathlib import Path

initializer = tf.keras.initializers.he_normal()  # Layer initializations

BASE_DIR = Path("/root/Master-Thesis2/code")  # Base path
MNIST_M_PATH = BASE_DIR / Path("data/keras_mnistm.pkl")
OFFICE_DS_PATH = BASE_DIR / Path("data/office/")
LOGS_DIR = BASE_DIR / Path("logs/")  # Logs path
MODEL_PATH = BASE_DIR / Path("model_data/")  # Model path
EVALUATION = BASE_DIR / Path("evaluation/")  # Evalaution plots path
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_COMBINATION = {
    "Amazon_to_Webcam": 1,
    "Amazon_to_DSLR": 2,
    "Webcam_to_Amazon": 3,
    "Webcam_to_DSLR": 4,
    "DSLR_to_Amazon": 5,
    "SynSigns_to_GTSRB": 6,
}

Architecture = {1: "ALexNet", 2: "Vgg16", 3: "Xception"}
