import pandas as pd
import seaborn as sn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(20)

# tf.keras.backend.clear_session()  # For easy reset of notebook state.
initializer = tf.keras.initializers.he_normal()


def test_accuracy(model, test_set, test_labels, batch=None, verbose=2):
    """[This method gives test_accuracy as on output]

    Arguments:
        model {[type]} -- [description]
        test_set {[type]} -- [description]
        test_labels {[type]} -- [description]

    Keyword Arguments:
        verbose {int} -- [description] (default: {2})
    """
    score = model.evaluate(test_set, test_labels, verbose=verbose, batch_size=batch)
    print(f"Test loss: {score[0]}, Test Accuracy: {score[1]}")


def evaluations(
    model,
    test_set,
    test_labels,
    batch=None,
    class_names_list=[],
    pass_pred_prob=False,
    y_pred=None,
):
    """[This method generates a Heat-Map, Confusion matrix and provides a count of
        Correct & Incorrect predictions]

    Arguments:
        model {[type]} -- [description]
        test_set {[type]} -- [description]
        test_labels {[type]} -- [description]

    Keyword Arguments:
        class_names_list {list} -- [description] (default: {[]})
    """
    if pass_pred_prob:
        y_prob = y_pred
    else:
        y_prob = model.predict(test_set, batch_size=batch)
    y_pred = np.argmax(y_prob, axis=1)
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=y_pred).numpy()
    class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    class_labels = [val + "-class" for val in class_labels]
    print(f"Total Test Cases: {con_mat.sum()}")
    temp_arr = np.eye(10)
    final_conf_mat = con_mat * temp_arr
    # print(final_conf_mat)
    correct_classifications = final_conf_mat.sum()
    incorrect_classifications = con_mat.sum() - correct_classifications
    print("Correct classifications:", int(correct_classifications))
    print("Incorrect classifications:", int(incorrect_classifications))
    pd.set_option("max_columns", None)

    con_mat_df = pd.DataFrame(con_mat, index=class_labels, columns=class_labels)

    print(con_mat_df)
    # Generating HeatMaps
    plt.figure(figsize=(12, 8))
    sn.heatmap(con_mat_df, annot=True, fmt="g")
