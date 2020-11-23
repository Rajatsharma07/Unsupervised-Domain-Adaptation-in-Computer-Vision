import tensorflow as tf

initializer = tf.keras.initializers.he_normal()  # Layer initializations

BASE_DIR = "/root/Master_Thesis/"
MNIST_M_PATH = BASE_DIR + "data/keras_mnistm.pkl"
LOGS_DIR = BASE_DIR + "logs/"
MODEL_PATH = BASE_DIR + "model_data/"
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASET_COMBINATION = {
    "MNIST_to_MNISTM": 1,
    "MNISTM_to_MNIST": 2,
    "GTSRB_to_SynSigns": 3,
}
