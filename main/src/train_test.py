import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import os
from pathlib import Path
import src.config as cn
from src.models import merged_model
from src.preprocessing import fetch_data
import src.utils as utils
from src.loss import CORAL, kl_divergence, coral_loss


def train_test(params):

    my_dir = (
        str(params["combination"])
        + "_"
        + str(params["architecture"])
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

    tf.compat.v1.logging.info(
        "Fetched the architecture function: " + cn.Architecture[params["architecture"]]
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

            model = merged_model(
                input_shape=(299, 299, 3),
                num_classes=31,
                lambda_loss=params["lambda_loss"],
                additional_loss=CORAL,
                prune=params["prune"],
                # freeze_upto=params["freeze_upto"],
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

        model = merged_model(
            input_shape=(299, 299, 3),
            num_classes=31,
            lambda_loss=params["lambda_loss"],
            additional_loss=CORAL,
            prune=params["prune"],
            # freeze_upto=params["freeze_upto"],
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
    # plot_model(model, os.path.join(log_dir, "Dual_Model.png"), show_shapes=True)
    # print_weights = tf.keras.callbacks.LambdaCallback(
    #     on_epoch_end=lambda batch, logs: print(
    #         model.get_layer("block5_conv2_2").get_weights()
    #     )
    # )
    # callbacks.append(print_weights)

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
        hist=hist, log_dir=log_dir, params=params,
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

    """ Evaluate on Target Dataset"""
    results = model.evaluate(ds_test)
    tf.compat.v1.logging.info(
        f"Test Set evaluation results for run {Path(log_dir).name} : Accuracy: {results[1]}, Loss: {results[0]}"
    )
    return model, hist, results
