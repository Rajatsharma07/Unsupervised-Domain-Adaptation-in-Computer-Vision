import tensorflow as tf
import matplotlib.pyplot as plt
import os
import logging
import pickle
from pathlib import Path
from tensorflow.keras.callbacks import CSVLogger
import datetime
import src.config as cn
import numpy as np


def define_logger(log_file):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # get TF logger
    log = logging.getLogger("tensorflow")
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def loss_accuracy_plots(
    hist,
    log_dir,
    params,
):
    accuracy = hist.history["accuracy"]
    val_accuracy = hist.history["val_accuracy"]
    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.plot(loss, "r", label="Training")
    plt.plot(val_loss, "r:", label="Validation")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(accuracy, "g", label="Training")
    plt.plot(val_accuracy, "g:", label="Validation")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plot_path = os.path.join(
        cn.EVALUATION, (Path(log_dir).parent).name, Path(log_dir).name
    )
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    tf.compat.v1.logging.info("Plots created at: " + plot_path)
    plot_path = os.path.join(plot_path, "Accuracy_Loss_Plots.png")
    plt.savefig(plot_path)
    plt.show()


def display_dataset(data, grayscale=True):
    """[This method visualizes the images present inside dataset, without label information]

    Args:
        data ([type]): [description]
        grayscale (bool, optional): [description]. Defaults to True.
    """
    plt.figure(figsize=(6, 6))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if grayscale:
            plt.imshow(data[i], cmap=plt.cm.binary)
        else:
            plt.imshow(data[i])
    plt.show()


def extract_mnist_m(mnistm_path):
    with open(mnistm_path, "rb") as f:
        mnistm_dataset = pickle.load(f, encoding="bytes")
    mnistmx_train = mnistm_dataset[b"train"]
    mnistmx_test = mnistm_dataset[b"test"]
    return mnistmx_train, mnistmx_test


def callbacks_fn(params, my_dir):

    callback_list = []
    """Tensorboard Callback """

    tb_logdir = os.path.join(cn.LOGS_DIR, my_dir)
    Path(tb_logdir).mkdir(parents=True, exist_ok=True)
    assert os.path.exists(tb_logdir), "tb_logdir doesn't exist"
    tb_logdir = os.path.join(
        tb_logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    log_dir = tb_logdir
    # file_writer = tf.summary.create_file_writer(tb_logdir + "/custom_evaluation")
    # file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(tb_logdir, histogram_freq=0)
    callback_list.append(tensorboard_callback)
    # print(f"\nTensorboard logs path: {tb_logdir}\n")
    tf.compat.v1.logging.info(f"Tensorboard logs path: {tb_logdir}")

    """CSV Logger Callback """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    csv = os.path.join(log_dir, "training_logs.csv")
    csv_logger = CSVLogger(
        csv,
        append=True,
        separator=";",
    )
    # print(f"\nModel CSV logs path: {csv}\n")
    tf.compat.v1.logging.info(f"Model CSV logs path: {csv}")
    callback_list.append(csv_logger)

    """Reduce LR Callback """
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.4, patience=2, min_lr=0.000001
    )
    callback_list.append(reduce_lr_callback)

    """Early Stopping Callback """
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        verbose=1,
        mode="auto",
    )
    callback_list.append(early_stopping_callback)

    """Checkpoint Callback """
    if params["save_weights"]:
        assert os.path.exists(cn.MODEL_PATH), "MODEL_PATH doesn't exist"
        checkpoint_path = os.path.join(
            cn.MODEL_PATH, (Path(log_dir).parent).name, Path(log_dir).name
        )
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        assert os.path.exists(checkpoint_path), "checkpoint_path doesn't exist"
        checkpoint_path = os.path.join(
            checkpoint_path,
            "weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
        )
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
            monitor="val_accuracy",
        )
        callback_list.append(cp_callback)
        # print(f"\nModel Checkpoint path: {checkpoint_path}\n")
        tf.compat.v1.logging.info(f"Model Checkpoint path: {checkpoint_path}")

    return callback_list, log_dir


def create_paths(path):
    all_image_paths = [str(path) for path in list(path.glob("*/*"))]
    label_names = sorted(item.name for item in path.glob("*/") if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    all_image_labels = [
        label_to_index[Path(path).parent.name] for path in all_image_paths
    ]

    return all_image_paths, all_image_labels


def shuffle_dataset(data_x, data_Y):
    # np.random.seed(seed_val)
    index_shuffled = np.arange(data_x.shape[0])
    np.random.shuffle(index_shuffled)
    data_x = np.array(data_x)
    data_x = data_x[index_shuffled]
    data_Y = (np.array(data_Y))[index_shuffled]
    return data_x, data_Y
