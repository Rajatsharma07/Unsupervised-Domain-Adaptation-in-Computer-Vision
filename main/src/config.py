import tensorflow as tf
from pathlib import Path
import tensorflow_model_optimization as tfmot

initializer = tf.keras.initializers.he_normal()  # Layer initializations

BASE_DIR = Path("/root/Master-Thesis")  # Base path
MNIST_M_PATH = BASE_DIR / Path("data/keras_mnistm.pkl")
LOGS_DIR = BASE_DIR / Path("logs/")  # Logs path
MODEL_PATH = BASE_DIR / Path("model_data/")  # Model path
EVALUATION = BASE_DIR / Path("evaluation/")  # Evalaution plots path
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_COMBINATION = {
    "MNIST_to_MNISTM": 1,
    "MNISTM_to_MNIST": 2,
    "SynSigns_to_GTSRB": 3,
    "GTSRB_to_SynSigns": 4,
}

Loss = {1: "CORAL", 2: "Other"}

pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
        0.20, 0, end_step=-1, frequency=1
    )
}
