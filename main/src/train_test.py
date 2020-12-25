from src.models import DeepCORAL
import tensorflow as tf
from tensorflow import keras
from src.preprocessing import fetch_data
import src.utils as utils
from tensorflow.keras.utils import plot_model
import os
import src.config as cn
from pathlib import Path


def train_test(params):

    my_dir = (
        str(params["combination"])
        + "_"
        + str(params["model_mode"])
        + "_"
        + str(params["lambda_loss"])
    )

    assert os.path.exists(cn.LOGS_DIR), "LOGS_DIR doesn't exist"
    experiment_logs_path = os.path.join(cn.LOGS_DIR, my_dir)
    Path(experiment_logs_path).mkdir(parents=True, exist_ok=True)
    utils.define_logger(os.path.join(experiment_logs_path, "experiments.log"))
    tf.compat.v1.logging.info("\n")
    tf.compat.v1.logging.info("Parameters: " + str(params))
    assert (
        params["mode"].lower() == "train_test"
    ), "change training mode to 'train_test'"

    tf.compat.v1.logging.info("Fetched the model model: " + str(params["model_mode"]))

    if params["use_multiGPU"]:
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        # Open a strategy scope.
        with strategy.scope():
            model = None
            tf.compat.v1.logging.info("Using Mutliple GPUs for training ...")
            tf.compat.v1.logging.info("Building the model ...")
            model = DeepCORAL(input_shape=(32, 32, 3), num_classes=10)

            """ Model Compilation """
            tf.compat.v1.logging.info("Compiling the model ...")
            model.compile(
                optimizer=keras.optimizers.Nadam(learning_rate=params["learning_rate"]),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )
    else:
        # Create model
        tf.compat.v1.logging.info("Building the model ...")
        model = None
        model = DeepCORAL(input_shape=(32, 32, 3), num_classes=10)

        """ Model Compilation """
        tf.compat.v1.logging.info("Compiling the model ...")
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=params["learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    """ Create callbacks """
    tf.compat.v1.logging.info("Creating the callbacks ...")
    callbacks, log_dir = utils.callbacks_fn(params, my_dir)
    plot_model(model, os.path.join(log_dir, "Dual_Model.png"), show_shapes=True)

    tf.compat.v1.logging.info("Calling data preprocessing pipeline...")
    ds_train, ds_val, ds_test = fetch_data(params)

    """ Model Training """
    tf.compat.v1.logging.info("Training Started....")
    hist = None
    hist = model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=params["epochs"],
        verbose=2,
        callbacks=callbacks,
        # batch_size=params["batch_size"],
    )
    tf.compat.v1.logging.info("Training finished....")

    """ Model Saving """
    if params["save_model"]:
        tf.compat.v1.logging.info("Saving the model...")
        model_path = os.path.join(
            cn.MODEL_PATH, (Path(log_dir).parent).name, Path(log_dir).name
        )
        Path(model_path).mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        tf.compat.v1.logging.info(f"Model successfully saved at: {model_path}")

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
    return model, hist, results
