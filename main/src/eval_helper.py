import pandas as pd
import seaborn as sn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import src.config as cn


def test_accuracy(model, test_set, test_labels, batch=None, verbose=2):
    """[This method gives test_accuracy as on output]

    Arguments:
        model {[keras.Model]} -- [Keras model]
        test_set {[type]} -- [test dataset]
        test_labels {[type]} -- [test labels]

    Keyword Arguments:
        verbose {int} -- [verbose access] (default: {2})
    """
    score = model.evaluate(test_set, test_labels, verbose=verbose, batch_size=batch)
    tf.compat.v1.logging.info(f"Test loss: {score[0]}, Test Accuracy: {score[1]}")


def evaluations(
    model,
    test_set,
    test_labels,
    log_dir,
    batch=None,
    class_names_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    pass_pred_prob=False,
    y_pred=None,
):
    """[This method generates a Heat-Map, Confusion matrix and provides a count of
        Correct & Incorrect predictions]

    Args:
        model ([type]): [model]
        test_set ([type]): [test dataset]
        test_labels ([type]): [test labels]
        log_dir ([str]): [path to save the figures]
        batch ([type], optional): [batch size]. Defaults to None.
        class_names_list (list, optional): [class labels]. Defaults to ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"].
        pass_pred_prob (bool, optional): [description]. Defaults to False.
        y_pred ([type], optional): [predicted labels]. Defaults to None.
    """
    if pass_pred_prob:
        y_prob = y_pred
    else:
        y_prob = model.predict(test_set, batch_size=batch)
    y_pred = np.argmax(y_prob, axis=1)
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=y_pred).numpy()
    class_labels = [val + "-class" for val in class_names_list]
    print(f"Total Test Cases: {con_mat.sum()}")
    temp_arr = np.eye(len(class_labels))
    final_conf_mat = con_mat * temp_arr
    correct_classifications = final_conf_mat.sum()
    incorrect_classifications = con_mat.sum() - correct_classifications
    tf.compat.v1.logging.info(
        f"Correct classifications: {int(correct_classifications)}"
    )
    tf.compat.v1.logging.info(
        f"Incorrect classifications: {int(incorrect_classifications)}"
    )
    pd.set_option("max_columns", None)
    con_mat_df = pd.DataFrame(con_mat, index=class_labels, columns=class_labels)
    # print(con_mat_df)
    # Generating HeatMaps
    plt.figure(figsize=(12, 8))
    sn.heatmap(con_mat_df, annot=True, fmt="g")
    plot_path = os.path.join(
        cn.EVALUATION, (Path(log_dir).parent).name, Path(log_dir).name
    )
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    plot_path = os.path.join(plot_path, "Heatmap.png")
    plt.savefig(plot_path)
    tf.compat.v1.logging.info(f"Evaluation plot saved at {plot_path}")
    plt.show()
