import pandas as pd
import seaborn as sn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import umap
import h5py
import config as cn
import os
from functools import reduce


def loss_accuracy_plots(
    accuracy, val_accuracy, loss, val_loss, combination, source_model, sample_seed
):
    """[This method generates an Accuracy-Loss graphs using MatplotLib]

    Arguments:
        accuracy {[float]} -- [description]
        val_accuracy {[float]} -- [description]
        loss {[float]} -- [description]
        val_loss {[float]} -- [description]
        combination {[str]} -- [description]
    """
    my_dir = combination + "_" + source_model + "_" + sample_seed
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
    if os.path.exists(cn.PLOTS_PATH):
        plot_path = os.path.join(cn.PLOTS_PATH, my_dir)
        os.makedirs(plot_path)
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
    tf.keras.backend.clear_session()  # For easy reset of notebook state.
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