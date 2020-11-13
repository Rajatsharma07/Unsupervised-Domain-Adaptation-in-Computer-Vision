import numpy as np
import tensorflow as tf

tf.random.set_seed(100)
initializer = tf.keras.initializers.he_normal()  # Layer initializations

BASE_DIR = "/root/Master_Thesis/"
seed_val = 121
np.random.seed(seed_val)
SOURCE_DS = None
TARGET_DS = None
SOURCE_MODEL = ""
MNIST_M_PATH = BASE_DIR + "input/mnist_m/keras_mnistm.pkl"
TENSORBOARD_DIR = BASE_DIR + "tensorboard_logs/"
LOGS_DIR = BASE_DIR + "csv_logs/"
MODEL_PATH = BASE_DIR + "model_data/"
PLOTS_PATH = BASE_DIR + "evaluation/"
EPOCHS = 5
BATCH_SIZE = 128
