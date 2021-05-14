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
import seaborn as sns
import bokeh
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import pickle
import datetime
import os


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


def test_accuracy(
    model,
    test_set=None,
    test_labels=None,
    batch=None,
    verbose=1,
    tf_dataset=None,
    is_tfdataset=False,
):
    """[This method gives test_accuracy as on output]

    Args:
        model ([keras.Model]): [Keras model]
        test_set ([type], optional): [test dataset]. Defaults to None.
        test_labels ([type], optional): [test labels]. Defaults to None.
        batch ([type], optional): [description]. Defaults to None.
        verbose (int, optional): [verbose access]. Defaults to 1.
        tf_dataset ([type], optional): [TFDS dataset]. Defaults to None.
        is_tfdataset (bool, optional): [Yes if TF data pipeline is used]. Defaults to False.
    """
    if is_tfdataset:
        loss, accuracy = model.evaluate(tf_dataset, verbose=verbose, batch_size=batch)
    else:
        loss, accuracy = model.evaluate(
            test_set, test_labels, verbose=verbose, batch_size=batch
        )
    tf.compat.v1.logging.info(f"Test loss: {loss}, Test Accuracy: {accuracy}")


def pruning_plots(
    rows,
    cols,
    count,
    x,
    y,
    names,
    figsize=(12, 5),
    x_divisions=1,
    y_divisions=10,
    achor_box=(0.7, 1.18),
    save_file="temp.pdf",
):
    plt.close("all")
    font = {"family": "serif", "weight": "normal", "size": 14}

    plt.rc("font", **font)
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = figsize
    count = list(range(count))
    for i, name in zip(count, names):
        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(
            x[i],
            y[i][0],
            color="blue",
            linestyle="dashed",
            marker="o",
            label="Best Accuracy",
        )
        ax.plot(x[i], y[i][1], color="green", marker="s", label="Pruned Accuracy")
        ax.set_facecolor("bisque")
        ax.set_xlabel("target_sparsity")
        ax.title.set_text(name)
        ax.set_ylabel("test accuracy", fontsize=12)
        if i != 4:
            plt.xticks(np.arange(min(x[i]), max(x[i]) + 1, x_divisions))
        else:
            plt.xticks(np.arange(0, 100, 8))
        plt.yticks(np.arange(0, max(y[i][0]) + 20, y_divisions))
    ax.legend(loc="upper center", bbox_to_anchor=achor_box)
    plt.subplots_adjust(hspace=0.35, wspace=None)
    fig.savefig(save_file, dpi=fig.dpi, bbox_inches="tight")
    return plt


if __name__ == "__main__":
    y1 = [
        55.5,
        56.6,
        56,
        56,
        55.22,
        60,
        55,
        57.86,
        53,
        49,
        50,
        51,
        55,
        60.1,
        55.84,
        57.35,
        50,
        51,
        33,
    ]
    y11 = [80] * len(y1)
    y2 = [
        65.26,
        68,
        67.26,
        64.9,
        63.25,
        61.64,
        65,
        61.24,
        65,
        63.45,
        65.26,
        62.85,
        69.47,
        69.27,
        65.46,
        69.27,
        64.65,
        60,
        46.78,
    ]
    y22 = [81] * len(y2)
    y3 = [32, 32.41, 31.13, 39, 34.3, 36.38, 36.63, 38.33, 34, 28.18]
    y33 = [63.50] * len(y3)

    y4 = [36.74, 30.38, 37.52, 37.52, 34.3, 39, 39.7, 39.65, 27.5, 19]
    y44 = [62.36] * len(y4)

    y5 = [
        66.08,
        68.57,
        70.2,
        66.26,
        69.2,
        69.32,
        71.3,
        70.37,
        73.30,
        71.5,
        69.62,
        70,
        68.23,
    ]
    y55 = [77.75] * len(y5)
    x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90]
    x3 = x1[9:]
    x5 = [2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90]

    pruning_plots(
        3,
        2,
        5,
        x=[x1, x1, x3, x3, x5],
        y=[[y11, y1], [y22, y2], [y33, y3], [y44, y4], [y55, y5]],
        x_divisions=8,
        figsize=(15, 16),
        achor_box=(0.75, 3),
        names=["A->W", "A->D", "W->A", "D->A", "S->G"],
        save_name="MBM_pruning.pdf",
    )
