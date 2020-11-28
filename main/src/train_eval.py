from src.source_model import resnet_50
from src.target_model import target_model
from src.combined_model import merged_network, callbacks_fn, Custom_Eval
import tensorflow as tf
from tensorflow import keras
from src.preprocessing import fetch_data
import src.utils as utils
from tensorflow.keras.utils import plot_model
import os
import src.config as cn
from pathlib import Path
from src.eval_helper import test_accuracy, evaluation_plots


def train_test(params):

    my_dir = (
        str(params["combination"])
        + "_"
        + str(params["source_model"])
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
        "Fetched the source model id: " + str(params["source_model"])
    )

    source_mdl = None
    target_mdl = None

    if params["use_multiGPU"]:
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))

        # Open a strategy scope.
        with strategy.scope():
            if params["source_model"] == 1:
                source_mdl = resnet_50(input_shape=(32, 32, 3), is_pretrained=False)
            elif params["source_model"] == 2:
                pass
            else:
                source_mdl = resnet_50(input_shape=(32, 32, 3), is_pretrained=True)

            # Call Target model
            tf.compat.v1.logging.info(
                "Fetched the target model: " + params["target_model"]
            )
            target_mdl = target_model(input_shape=(32, 32, 3))

            tf.compat.v1.logging.info("Using Mutliple GPUs for training ...")
            # Create Combined model
            tf.compat.v1.logging.info("Building the combined model ...")
            model = None
            model = merged_network(
                input_shape=(32, 32, 3),
                source_model=source_mdl,
                target_model=target_mdl,
                lambda_loss=params["lambda_loss"],
            )

            """ Model Compilation """
            tf.compat.v1.logging.info("Compiling the combined model ...")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )
    else:
        if params["source_model"] == 1:
            source_mdl = resnet_50(input_shape=(32, 32, 3), is_pretrained=False)
        elif params["source_model"] == 2:
            pass
        else:
            source_mdl = resnet_50(input_shape=(32, 32, 3), is_pretrained=True)

        # Call Target model
        tf.compat.v1.logging.info("Fetched the target model: " + params["target_model"])
        target_mdl = target_model(input_shape=(32, 32, 3))

        # Create Combined model
        tf.compat.v1.logging.info("Building the combined model ...")
        model = None
        model = merged_network(
            input_shape=(32, 32, 3),
            source_model=source_mdl,
            target_model=target_mdl,
            lambda_loss=params["lambda_loss"],
        )

        """ Model Compilation """
        tf.compat.v1.logging.info("Compiling the combined model ...")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    """ Create callbacks """
    tf.compat.v1.logging.info("Creating the callbacks ...")
    callbacks, log_dir = callbacks_fn(params, my_dir)

    plot_model(source_mdl, os.path.join(log_dir, "source_model.png"), show_shapes=True)
    plot_model(target_mdl, os.path.join(log_dir, "target_model.png"), show_shapes=True)
    plot_model(model, os.path.join(log_dir, "merged_model.png"), show_shapes=True)

    tf.compat.v1.logging.info("Calling data preprocessing pipeline...")
    ds_train, ds_val, ds_test = fetch_data(params)

    """ Custom Evauation Callback """
    tf.compat.v1.logging.info("Creating custom evaluation callback...")
    custom_eval = Custom_Eval(ds_val)
    callbacks[:0] = [custom_eval]

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


def evaluate(
    model_path,
    test_set,
    test_labels,
    log_dir,
    params,
    class_names_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    batch_size=None,
):
    tf.compat.v1.logging.info("Starting Evaluation...")
    test_accuracy(
        test_set=test_set, test_labels=test_labels, model=model_path, batch=batch_size
    )
    evaluation_plots(
        model=model_path,
        test_set=test_set,
        test_labels=test_labels,
        log_dir=log_dir,
        batch=batch_size,
        class_names_list=class_names_list,
        params=params,
    )
