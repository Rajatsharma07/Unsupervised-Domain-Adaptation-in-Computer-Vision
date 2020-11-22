import config as cn
from src.preprocessing import get_mnist, get_mnist_m, shuffle_dataset
from src.source_model import source_resnet
from src.target_model import target_model
from src.combined_model import merged_network, callbacks_fn, Custom_Eval
import tensorflow as tf
from tensorflow import keras
import src.utils as utils
import src.eval_helper as evals
import os
import argparse
from tensorboard.plugins.hparams import api as hp


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    tf.compat.v1.logging.info("Building the model ...")

    source_mdl = None
    source_mdl = source_resnet((32, 32, 3))

    # Call Target model
    target_mdl = None
    target_mdl = target_model(input_shape=(32, 32, 3))

    # Create Combined model
    model = None
    model = merged_network(
        (32, 32, 3), source_model=source_mdl, target_model=target_mdl, percent=0.7
    )

    """ Compilation and Fit"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Create callbacks
    (
        csv_logger,
        cp_callback,
        tensorboard_callback,
        reduce_lr_callback,
        logdir,
    ) = callbacks_fn(
        combination=args.combination,
        source_model=args.source_model,
        sample_seed=args.sample_seed,
    )

    # Custom Evauation Callback
    custom_eval = Custom_Eval((mnistmx_test, mnisty_test_or))

    hist = None
    hist = model.fit(
        x=[mnistx_train, mnistmx_train],
        y=mnisty_train,
        validation_data=(
            [mnistmx_train, mnistmx_train],
            mnisty_train_or,
        ),
        epochs=cn.EPOCHS,
        verbose=1,
        callbacks=[
            cp_callback,
            tensorboard_callback,
            reduce_lr_callback,
            custom_eval,
            csv_logger,
        ],
        batch_size=cn.BATCH_SIZE,
    )