import pandas as pd
import seaborn as sn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import src.config as cn


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


def evaluation_plots(
    model,
    test_labels,
    log_dir,
    class_names_list,
    params,
    test_set=None,
    batch=None,
    tf_dataset=None,
    is_tfdataset=False,
):
    """[This method generates a Heat-Map, Confusion matrix and provides a count of Correct & Incorrect predictions]

    Args:
        model ([keras.Model]): [description]
        test_set ([type]): [test_set]. Defaults to None.
        test_labels ([type]): [test_labels]
        log_dir ([str]): [log direcory]
        class_names_list ([type]): [class labels]
        params ([dict]): [Argparse dictionary]
        batch ([int], optional): [batch size]. Defaults to None.
        tf_dataset ([type], optional): [TFDS dataset]. Defaults to None.
        is_tfdataset (bool, optional): [Yes if TF data pipeline is used]. Defaults to False.
    """
    if is_tfdataset:
        y_pred = model.predict(tf_dataset, batch_size=batch)
    else:
        y_pred = model.predict(test_set, batch_size=batch)
    # y_pred = np.argmax(y_prob, axis=1)
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=y_pred).numpy()
    class_labels = [val + "-class" for val in class_names_list]
    tf.compat.v1.logging.info(f"Total Test Cases: {con_mat.sum()}")
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
