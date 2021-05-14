import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import os
from tensorflow.python.ops.gen_batch_ops import batch
import tensorflow_model_optimization as tfmot
from pathlib import Path
import src.config as cn
from src.models import get_model
from src.preprocessing import fetch_data
import src.utils as utils
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def train_test(params):

    my_dir = (
        str(cn.DATASET_COMBINATION[params["combination"]])
        + "_"
        + str(params["architecture"])
        + "_"
        + str(params["loss_function"])
        + "_"
        + str(params["lambda_loss"])
    )

    if not params["technique"]:
        my_dir = my_dir + "_Original"

    if params["prune"]:
        tf.compat.v1.logging.info("Pruning is activated")
        my_dir = my_dir + "_" + str(params["prune_val"])

    assert os.path.exists(cn.LOGS_DIR), "LOGS_DIR doesn't exist"
    experiment_logs_path = os.path.join(cn.LOGS_DIR, my_dir)
    Path(experiment_logs_path).mkdir(parents=True, exist_ok=True)
    utils.define_logger(os.path.join(experiment_logs_path, "experiments.log"))
    tf.compat.v1.logging.info("\n")
    tf.compat.v1.logging.info("Parameters: " + str(params))
    assert (
        params["mode"].lower() == "train_test"
    ), "change training mode to 'train_test'"

    tf.compat.v1.logging.info(
        "Fetched the architecture function: " + params["architecture"]
    )

    if params["use_multiGPU"]:
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        # Open a strategy scope.
        with strategy.scope():
            model = None
            tf.compat.v1.logging.info("Using Mutliple GPUs for training ...")
            tf.compat.v1.logging.info("Building the model ...")

            model = get_model(
                input_shape=params["input_shape"],
                num_classes=params["output_classes"],
                lambda_loss=params["lambda_loss"],
                additional_loss=params["loss_function"],
                prune=params["prune"],
                prune_val=params["prune_val"],
                technique=params["technique"],
            )

            # print(model.summary())
            """ Model Compilation """
            tf.compat.v1.logging.info("Compiling the model ...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )
    else:
        # Create model
        tf.compat.v1.logging.info("Building the model ...")

        model = None

        model = get_model(
            input_shape=params["input_shape"],
            num_classes=params["output_classes"],
            lambda_loss=params["lambda_loss"],
            additional_loss=params["loss_function"],
            prune=params["prune"],
            prune_val=params["prune_val"],
            technique=params["technique"],
        )

        # print(model.summary())
        """ Model Compilation """
        tf.compat.v1.logging.info("Compiling the model ...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    """ Create callbacks """
    tf.compat.v1.logging.info("Creating the callbacks ...")
    callbacks, log_dir = utils.callbacks_fn(params, my_dir)

    tf.compat.v1.logging.info("Calling data preprocessing pipeline...")
    ds_train, ds_test = fetch_data(params)

    """ Model Training """
    tf.compat.v1.logging.info("Training Started....")

    hist = None
    hist = model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=params["epochs"],
        verbose=1,
        callbacks=callbacks,
    )
    tf.compat.v1.logging.info("Training finished....")

    """ Plotting """
    tf.compat.v1.logging.info("Creating accuracy & loss plots...")
    utils.loss_accuracy_plots(
        hist=hist,
        log_dir=log_dir,
        params=params,
    )

    """ Evaluate on Target Dataset"""
    results = model.evaluate(ds_test)
    tf.compat.v1.logging.info(
        f"Test Set evaluation results for run {Path(log_dir).name} : Accuracy: {results[1]}, Loss: {results[0]}"
    )

    """ Model Saving """
    if params["save_model"]:
        tf.compat.v1.logging.info("Saving the model...")
        model_path = os.path.join(
            cn.MODEL_PATH, (Path(log_dir).parent).name, Path(log_dir).name
        )
        Path(model_path).mkdir(parents=True, exist_ok=True)
        model.save(os.path.join(model_path, "model"))
        tf.compat.v1.logging.info(f"Model successfully saved at: {model_path}")

    """ Pruned Model Saving """
    if params["prune"]:
        model_for_export = tfmot.sparsity.keras.strip_pruning(model)
        tf.compat.v1.logging.info(f"Pruned Model summary: {model_for_export.summary()}")

        tf.compat.v1.logging.info("Saving Pruned Model...")
        model_path = os.path.join(
            cn.MODEL_PATH, (Path(log_dir).parent).name, Path(log_dir).name
        )
        Path(model_path).mkdir(parents=True, exist_ok=True)
        model_for_export.save(os.path.join(model_path, "pruned_model"))
        tf.compat.v1.logging.info(f"Pruned Model successfully saved at: {model_path}")

        tf.compat.v1.logging.info(
            "Size of gzipped pruned model without stripping: %.2f bytes"
            % (utils.get_gzipped_model_size(model))
        )

        tf.compat.v1.logging.info(
            "Size of gzipped pruned model with stripping: %.2f bytes"
            % (utils.get_gzipped_model_size(model_for_export))
        )

    return model, hist, results


def evaluate(model_path, class_names_list, params, save_file, figsize=(17.5, 14)):
    """[This method generates a Heat-Map, Confusion matrix and provides a count of Correct & Incorrect predictions]

    Args:
        model_path ([keras.Model]): [path of trained keras model]
        class_names_list ([type]): [class labels]
        params ([dict]): [Argparse dictionary]
    """
    plt.close("all")
    font = {"family": "serif", "weight": "normal", "size": 13}

    plt.rc("font", **font)
    plt.xticks(rotation=90)
    # Featch the test dataset
    ds_train, ds_test = fetch_data(params)

    # ds_test_labels = ds_test.map(lambda x, y: y)
    true_categories = tf.concat([y for x, y in ds_test.unbatch()], axis=0)

    # Loading the trained model
    model = keras.models.load_model(model_path)

    # Predict the classes on the test dataset
    y_pred = model.predict(ds_test.unbatch().batch(1))
    y_pred = y_pred.argmax(axis=-1)
    # y_pred = np.argmax(y_prob, axis=1)

    con_mat = tf.math.confusion_matrix(
        labels=true_categories, predictions=tf.squeeze(y_pred)
    ).numpy()

    # class_labels = [val + "-class" for val in class_names_list]

    tf.compat.v1.logging.info(f"Total Test Cases: {con_mat.sum()}")
    temp_arr = np.eye(len(class_names_list))
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
    con_mat_df = pd.DataFrame(con_mat, index=class_names_list, columns=class_names_list)

    # Generating HeatMaps
    plt.rcParams["figure.figsize"] = figsize
    sn.heatmap(con_mat_df, cmap="YlGnBu")

    plot_path = os.path.join(model_path, save_file)
    # plt.savefig(plot_path)
    plt.savefig(plot_path)
    tf.compat.v1.logging.info(f"Evaluation plot saved at {plot_path}")
    plt.show()
    return plt


if __name__ == "__main__":
    model_path = "/root/Master-Thesis3/code/model_data/5_Xception_CORAL_0.75_Original/20210307-181912/model"
    plot = evaluate(model_path, params)