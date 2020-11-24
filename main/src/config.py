import tensorflow as tf
from pathlib import Path

initializer = tf.keras.initializers.he_normal()  # Layer initializations

BASE_DIR = Path("/root/Master-Thesis")
MNIST_M_PATH = BASE_DIR / Path("data/keras_mnistm.pkl")
LOGS_DIR = BASE_DIR / Path("logs/")
MODEL_PATH = BASE_DIR / Path("model_data/")
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_COMBINATION = {
    "MNIST_to_MNISTM": 1,
    "MNISTM_to_MNIST": 2,
    "GTSRB_to_SynSigns": 3,
}
