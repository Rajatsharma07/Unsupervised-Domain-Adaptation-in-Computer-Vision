from src.source_model import source_resnet
from src.target_model import target_model
from src.combined_model import merged_network, callbacks_fn, Custom_Eval
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from src.preprocessing import fetch_data
import src.utils as utils
import os
from tensorflow.keras.utils import plot_model


def train(params):
    tf.compat.v1.logging.info("\n")
    tf.compat.v1.logging.info("Parameters: " + str(params))
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    tf.compat.v1.logging.info("Fetched the source model: " + params["source_model"])
    source_mdl = None
    source_mdl = source_resnet((32, 32, 3))

    # Call Target model
    tf.compat.v1.logging.info("Fetched the target model: " + params["target_model"])
    target_mdl = None
    target_mdl = target_model(input_shape=(32, 32, 3))

    # Create Combined model
    tf.compat.v1.logging.info("Building the combined model ...")
    model = None
    model = merged_network(
        (32, 32, 3), source_model=source_mdl, target_model=target_mdl, percent=0.7
    )

    """ Compilation and Fit"""
    tf.compat.v1.logging.info("Compiling the combined model ...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Create callbacks
    tf.compat.v1.logging.info("Creating the callbacks ...")
    callbacks, log_dir = callbacks_fn(params)

    # plot_model(source_mdl, os.path.join(log_dir, "source_model.png"), show_shapes=True)
    # plot_model(target_mdl, os.path.join(log_dir, "target_model.png"), show_shapes=True)
    # plot_model(model, os.path.join(log_dir, "merged_model.png"), show_shapes=True)

    tf.compat.v1.logging.info("Calling data preprocessing pipeline...")
    ds_train, ds_custom_val, ds_test = fetch_data(params)

    # Custom Evauation Callback
    tf.compat.v1.logging.info("Creating custom evaluation callback...")
    custom_eval = Custom_Eval(ds_custom_val)
    callbacks[:0] = [custom_eval]

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

    # Plotting
    tf.compat.v1.logging.info("Creating accuracy & loss plots...")
    utils.loss_accuracy_plots(
        hist.history["accuracy"],
        hist.history["val_accuracy"],
        hist.history["loss"],
        hist.history["val_loss"],
        params,
    )

    if params["save_model"]:
        tf.compat.v1.logging.info("Saving the model...")
        model.save(log_dir)
