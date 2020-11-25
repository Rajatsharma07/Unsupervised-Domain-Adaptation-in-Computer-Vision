import pandas as pd
import seaborn as sn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import umap
import h5py
import src.config as cn
import os
import io
import sklearn.metrics
from functools import reduce
import logging
import pickle
from pathlib import Path


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
    # plot_path = os.path.join(cn.EVALUATION, my_dir, subdirectory)
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    plot_path = os.path.join(plot_path, "Accuracy_Loss_Plots.png")
    plt.savefig(plot_path)
    plt.show()


def display_dataset(data, grayscale=True):
    """[This method visualizes the images present inside dataset]

    Arguments:
        data {[type]} -- [description]
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


def plot_UMAP(
    input_data,
    input_labels,
    n_neighbors=5,
    min_dist=0.3,
    metric="correlation",
    alpha=0.6,
    height=9,
    palette="muted",
) -> None:
    """[summary]

    Arguments:
        input_data {[type]} -- [description]
        input_labels {[type]} -- [description]

    Keyword Arguments:
        n_neighbors {int} -- [description] (default: {5})
        min_dist {float} -- [description] (default: {0.3})
        metric {str} -- [description] (default: {"correlation"})
        alpha {float} -- [description] (default: {0.6})
        height {int} -- [description] (default: {9})
        palette {str} -- [description] (default: {"muted"})
    """
    embedding = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=5
    ).fit_transform(input_data)
    print("shape of umap_reduced.shape = ", embedding.shape)
    # attaching the label for each 2-d data point
    # creating a new data fram which help us in ploting the result data
    umap_data = np.vstack((embedding.T, input_labels)).T
    umap_df = pd.DataFrame(data=umap_data, columns=("Dim_1", "Dim_2", "Digits"))
    umap_df["Digits"] = umap_df["Digits"].astype(int)
    sn.set(style="whitegrid")
    plt.style.use("dark_background")
    sn.FacetGrid(umap_df, hue="Digits", height=height, palette=palette).map(
        plt.scatter, "Dim_1", "Dim_2", alpha=alpha
    ).add_legend()


def model_layers(model) -> None:
    """[Shows the layers inside a model and confirm it's trainable or not]

    Args:
        model ([Keras mode]): [created keras model]
    """
    for layers in model.layers:
        print(f"Layer Name: {layers.name} \t Trainable: {layers.trainable} ")


def hdf5(path, data_key="data", target_key="target", flatten=True):
    """
    loads data from hdf5:
    - hdf5 should have 'train' and 'test' groups
    - each group should have 'data' and 'target' dataset or spcify the key
    - flatten means to flatten images N * (C * H * W) as N * D array
    """
    with h5py.File(path, "r") as hf:
        train = hf.get("train")
        X_tr = train.get(data_key)[:]
        y_tr = train.get(target_key)[:]
        test = hf.get("test")
        X_te = test.get(data_key)[:]
        y_te = test.get(target_key)[:]
        if flatten:
            X_tr = X_tr.reshape(
                X_tr.shape[0], reduce(lambda a, b: a * b, X_tr.shape[1:])
            )
            X_te = X_te.reshape(
                X_te.shape[0], reduce(lambda a, b: a * b, X_te.shape[1:])
            )
    return X_tr, y_tr, X_te, y_te


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(data, labels, class_names):
    # Data should be in (BATCH_SIZE, H, W, C)
    assert data.ndim == 4

    figure = plt.figure(figsize=(10, 10))
    num_images = data.shape[0]
    size = int(np.ceil(np.sqrt(num_images)))

    for i in range(data.shape[0]):
        plt.subplot(size, size, i + 1, title=class_names[labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # if grayscale
        if data.shape[3] == 1:
            plt.imshow(data[i], cmap=plt.cm.binary)

        else:
            plt.imshow(data[i])

    return figure


def get_confusion_matrix(y_labels, logits, class_names):
    preds = np.argmax(logits, axis=1)
    cm = sklearn.metrics.confusion_matrix(
        y_labels,
        preds,
        labels=np.arange(len(class_names)),
    )

    return cm


def plot_confusion_matrix(cm, class_names):
    size = len(class_names)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    indices = np.arange(len(class_names))
    plt.xticks(indices, class_names, rotation=45)
    plt.yticks(indices, class_names)

    # Normalize Confusion Matrix
    cm = np.around(
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis],
        decimals=3,
    )

    threshold = cm.max() / 2.0
    for i in range(size):
        for j in range(size):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(
                i,
                j,
                cm[i, j],
                horizontalalignment="center",
                color=color,
            )

    plt.tight_layout()
    plt.xlabel("True Label")
    plt.ylabel("Predicted label")

    cm_image = plot_to_image(figure)
    return cm_image
