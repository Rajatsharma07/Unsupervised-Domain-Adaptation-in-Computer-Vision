import tensorflow as tf
from pathlib import Path
import tensorflow_model_optimization as tfmot

initializer = tf.keras.initializers.he_normal()  # Layer initializations

BASE_DIR = Path("/root/Master-Thesis")  # Base path
MNIST_M_PATH = BASE_DIR / Path("data/keras_mnistm.pkl")
OFFICE_DS_PATH = BASE_DIR / Path("data/office31/")
LOGS_DIR = BASE_DIR / Path("logs/")  # Logs path
MODEL_PATH = BASE_DIR / Path("model_data/")  # Model path
EVALUATION = BASE_DIR / Path("evaluation/")  # Evalaution plots path
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_COMBINATION = {
    "MNIST_to_MNISTM": 1,
    "MNISTM_to_MNIST": 2,
    "Amazon_to_Webcam": 3,
    "Amazon_to_DSLR": 4,
    "Webcam_to_Amazon": 5,
    "Webcam_to_DSLR": 6,
    "DSLR_to_Amazon": 7,
    "DSLR_to+Webcam": 8,
    "SynSigns_to_GTSRB": 9,
    "GTSRB_to_SynSigns": 10,
}

Loss = {1: "CORAL", 2: "Other"}

pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
        0.20, 0, end_step=-1, frequency=1
    )
}
