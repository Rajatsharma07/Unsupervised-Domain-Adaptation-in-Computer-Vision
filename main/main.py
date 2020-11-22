from src.preprocessing import get_mnist, get_mnist_m, shuffle_dataset
import config as cn
from src.source_model import source_resnet
from src.target_model import target_model
from src.model_dispatcher import merged_network, callbacks_fn, Custom_Eval
import tensorflow as tf
from tensorflow import keras
import src.utils as utils
import src.evals as evals
import os
import argparse
from tensorboard.plugins.hparams import api as hp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cm",
        "--combination",
        type=str,
        required=True,
        help="pass dataset combination string",
    )
    parser.add_argument(
        "sm",
        "--source_model",
        type=str,
        required=True,
        help="pass source model's method name",
    )
    parser.add_argument(
        "ss",
        "--sample_seed",
        type=int,
        required=True,
        help="pass the seed value for shuffling target dataset",
    )
    args = parser.parse_args()
    params = vars(args)
    print(params)

    # assert params["mode"], "mode is required. train, test or eval option"
    # assert params["mode"] in [
    #     "train",
    #     "test",
    #     "eval",
    # ], "The mode must be train , test or eval"
    # assert os.path.exists(params["data_dir"]), "data_dir doesn't exist"
    # assert os.path.isfile(params["vocab_path"]), "vocab_path doesn't exist"

    """ Fetch Source & Target Dataset"""
    mnistx_train_or, mnisty_train_or, mnistx_test_or, mnisty_test_or = get_mnist()

    mnistx_train, mnisty_train = shuffle_dataset(
        mnistx_train_or, mnisty_train_or, cn.seed_val
    )
    # mnistx_test, mnisty_test = shuffle_dataset(mnistx_test, mnisty_test, seed_val)

    mnistmx_train, mnistmx_test = get_mnist_m(cn.MNIST_M_PATH)

    # mnistmx_train, mnistmy_train = shuffle_dataset(
    #     mnistmx_train, mnisty_train, cn.seed_val
    # )

    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # # Open a strategy scope.
    # with strategy.scope():
    # Call Source model
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

    """ Evaluation"""
    utils.loss_accuracy_plots(
        hist.history["accuracy"],
        hist.history["val_accuracy"],
        hist.history["loss"],
        hist.history["val_loss"],
        combination=args.combination,
        source_model=args.source_model,
        sample_seed=args.sample_seed,
    )

    evals.test_accuracy(model, [mnistmx_test, mnistmx_test], mnisty_test_or)

    evals.test_accuracy(model, [mnistmx_test, mnistmx_test], mnisty_test_or)


if __name__ == "__main__":
    main()
