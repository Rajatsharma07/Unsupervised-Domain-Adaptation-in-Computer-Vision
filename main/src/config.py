import tensorflow as tf

tf.random.set_seed(100)
initializer = tf.keras.initializers.he_normal()  # Layer initializations

BASE_DIR = "/root/Master_Thesis/"
MNIST_M_PATH = BASE_DIR + "data/keras_mnistm.pkl"
TENSORBOARD_DIR = BASE_DIR + "tensorboard_logs/"
LOGS_DIR = BASE_DIR + "csv_logs/"
MODEL_PATH = BASE_DIR + "model_data/"
PLOTS_PATH = BASE_DIR + "evaluation/"
EPOCHS = 5
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
