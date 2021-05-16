import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import os
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
from sklearn.metrics import classification_report, roc_auc_score
import pandas
from scipy.special import softmax


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


def evaluate(model_path, params, figsize=(20, 15)):
    """[This method generates a Heat-Map, providesConfusion matrix and provides
    classification report and AUC score]

    Args:
        model_path ([keras.Model]): [path of trained keras model]
        params ([dict]): [Argparse dictionary]
        figsize (tuple): [Plot figure size]
    """
    plt.close("all")
    font = {"family": "serif", "weight": "bold", "size": 10}
    plt.rc("font", **font)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)

    files_path = os.path.join(cn.BASE_DIR, (Path(model_path).parent).name)
    Path(files_path).mkdir(parents=True, exist_ok=True)

    utils.define_logger(os.path.join(files_path, "evaluations.log"))

    tf.compat.v1.logging.info("Fetch the test dataset ...")
    _, ds_test = fetch_data(params)

    true_categories = tf.concat([y for x, y in ds_test], axis=0)
    np.save(os.path.join(files_path, "y_true"), true_categories.numpy())

    tf.compat.v1.logging.info("Loading the trained model ...")
    model = keras.models.load_model(model_path)

    tf.compat.v1.logging.info("Recompiling the model ...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    tf.compat.v1.logging.info("Predict the classes on the test dataset ...")
    y_pred = model.predict(ds_test)
    np.save(os.path.join(files_path, "y_prob"), y_pred)

    predicted_categories = tf.argmax(y_pred, axis=1)
    np.save(
        os.path.join(files_path, "predicted_categories"), predicted_categories.numpy()
    )

    tf.compat.v1.logging.info("Generating Classification Report ...")
    report = classification_report(
        true_categories, predicted_categories, output_dict=True
    )

    df = pandas.DataFrame(report).transpose()
    df.to_excel(os.path.join(files_path, "report.xlsx"))
    df = df.sort_values("f1-score")
    df.to_excel(os.path.join(files_path, "sorted.xlsx"))

    score = roc_auc_score(true_categories, softmax(y_pred, axis=1), multi_class="ovr")
    tf.compat.v1.logging.info("AUC score: " + str(score))

    conf_matrix = tf.math.confusion_matrix(
        labels=true_categories,
        predictions=predicted_categories,
    ).numpy()
    np.save(os.path.join(files_path, "conf_matrix"), conf_matrix)

    tf.compat.v1.logging.info("Generating Heatmaps ...")
    plt.rcParams["figure.figsize"] = figsize
    figure = sn.heatmap(conf_matrix, xticklabels=1, yticklabels=1)
    plt.xlabel("Actual Predictions", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Classes", fontsize=11, fontweight="bold")

    plot_path = os.path.join(Path(files_path), "Heatmap.pdf")
    plt.savefig(plot_path)
    figure = figure.get_figure()
    figure.savefig(plot_path)
    tf.compat.v1.logging.info(f"Evaluation plot saved at {plot_path}")
    plt.show()

    return plt
