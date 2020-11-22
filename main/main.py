from src.preprocessing import get_mnist, get_mnist_m, shuffle_dataset
import config as cn
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
from train_test_eval import train

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combination",
        type=str,
        required=True,
        help="pass dataset combination string",
    )
    parser.add_argument(
        "--source_model",
        type=str,
        required=True,
        help="pass source model's method name",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        required=True,
        help="pass the seed value for shuffling target dataset",
    )

    parser.add_argument("--batch_size", default=64, help="batch size", type=int)

    parser.add_argument(
        "--learning_rate", default=0.001, help="Learning rate", type=float
    )

    parser.add_argument(
        "--mode", help="training, eval or test options", default="", type=str
    )

    parser.add_argument(
        "--checkpoint_dir", help="Checkpoint directory", default="", type=str
    )
    parser.add_argument(
        "--test_save_dir",
        help="Directory in which we store the results",
        default="",
        type=str,
    )
    parser.add_argument("--data_dir", help="Data Folder", default="", type=str)

    parser.add_argument(
        "--log_file",
        help="File in which to redirect console outputs",
        default="",
        type=str,
    )
    args = parser.parse_args()
    params = vars(args)
    print(params)

    """ Fetch Source & Target Dataset"""
    mnistx_train_or, mnisty_train_or, mnistx_test_or, mnisty_test_or = get_mnist()

    mnistx_train, mnisty_train = shuffle_dataset(
        mnistx_train_or, mnisty_train_or, cn.seed_val
    )
    # mnistx_test, mnisty_test = shuffle_dataset(mnistx_test, mnisty_test, seed_val)

    mnistmx_train, mnistmx_test = get_mnist_m(cn.MNIST_M_PATH)

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
