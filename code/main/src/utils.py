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
import tensorflow_model_optimization as tfmot
import tempfile


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


def loss_accuracy_plots(hist, log_dir, params):

    font = {"family": "serif", "weight": "normal", "size": 12}
    accuracy = hist.history["accuracy"]
    val_accuracy = hist.history["val_accuracy"]
    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]

    plt.rc("font", **font)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(accuracy, color="green", marker="o", label="Train Accuracy")
    ax.plot(
        val_accuracy,
        linestyle="--",
        color="green",
        marker="d",
        alpha=0.5,
        label="Test Accuracy",
    )

    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_facecolor("bisque")
    ax.legend(loc="upper center", bbox_to_anchor=(0.7, 1.17))

    ax2 = ax.twinx()
    ax2.plot(loss, color="blue", marker="o", label="Train Loss")
    ax2.plot(
        val_loss, linestyle="--", color="blue", marker="d", alpha=0.5, label="Test Loss"
    )
    ax2.set_ylabel("Loss", fontsize=14)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.3, 1.17))
    # plt.xticks(np.arange(min(x), max(x) + 1, divisions))
    plot_path = os.path.join(
        cn.EVALUATION, (Path(log_dir).parent).name, Path(log_dir).name
    )

    Path(plot_path).mkdir(parents=True, exist_ok=True)
    tf.compat.v1.logging.info("Plots created at: " + plot_path)
    plot_path = os.path.join(plot_path, "Accuracy_Loss_Plots.png")
    plt.savefig(plot_path)
    plt.show()


def display_dataset(data, label, grayscale=True):
    plt.figure(figsize=(6, 6))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title(f"{label}")
        if grayscale:
            plt.imshow(data[i], cmap=plt.cm.binary)
        else:
            plt.imshow(data[i])
    plt.show()


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
    if not params["prune"]:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            tb_logdir, histogram_freq=1
        )
        callback_list.append(tensorboard_callback)
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
        monitor="val_accuracy", factor=0.4, patience=4, min_lr=0.0000001
    )
    callback_list.append(reduce_lr_callback)

    """Early Stopping Callback """
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
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

    """Pruning Callback """
    if params["prune"]:
        tf.compat.v1.logging.info(f"Tensorboard logs path: {tb_logdir}")
        callback_list.append(tfmot.sparsity.keras.UpdatePruningStep())
        # Log sparsity and other metrics in Tensorboard.
        callback_list.append(tfmot.sparsity.keras.PruningSummaries(log_dir=tb_logdir))

    return callback_list, log_dir


def get_gzipped_model_size(model):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, keras_file = tempfile.mkstemp(".h5")
    model.save(keras_file, include_optimizer=False)

    _, zipped_file = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zipped_file, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)

    return os.path.getsize(zipped_file)
