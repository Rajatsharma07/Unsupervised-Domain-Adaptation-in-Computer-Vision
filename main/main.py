import config as cn
import tensorflow as tf
import src.utils as utils
import src.eval_helper as evals
import os
import argparse
from src.train_test_eval import train

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combination",
        type=int,
        required=True,
        help="pass experiment combination, see config file",
    )

    parser.add_argument(
        "--source_model",
        type=str,
        required=True,
        help="pass source model's method name",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        help="pass target model's method name",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        required=True,
        default=500,
        help="pass the seed value for shuffling target dataset",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=32,
        help="pass image resizing dimension",
    )

    parser.add_argument("--batch_size", default=64, help="batch size", type=int)

    parser.add_argument(
        "--learning_rate", default=0.001, help="Learning rate", type=float
    )

    parser.add_argument(
        "--mode", help="train, eval or test options", default="", type=str
    )

    # parser.add_argument(
    #     "--checkpoint_dir", help="Checkpoint directory", default="", type=str
    # )
    # parser.add_argument(
    #     "--test_save_dir",
    #     help="Directory in which we store the results",
    #     default="",
    #     type=str,
    # )
    parser.add_argument(
        "--source_data_dir", help="Source Data path", default="", type=str
    )
    parser.add_argument(
        "--target_data_dir", help="Target Data path", default="", type=str
    )

    parser.add_argument(
        "--log_file",
        help="File in which to redirect console outputs",
        default=os.path.join(cn.LOGS_DIR, "main.log"),
        type=str,
    )
    parser.add_argument("--epochs", default=5, help="Epochs", type=int)

    parser.add_argument(
        "--save_weights",
        default=False,
        help="If yes, weights will be saved, otherwise not",
        type=bool,
    )

    parser.add_argument(
        "--save_model",
        default=False,
        help="If yes, model will be saved, otherwise not",
        type=bool,
    )

    args = parser.parse_args()
    params = vars(args)
    print(params)

    utils.define_logger(params["log_file"])

    assert params[
        "mode"
    ], "mode is required, please provide either train, test or eval option"

    assert params["mode"] in [
        "train",
        "test",
        "eval",
    ], "The mode must be train , test or eval"

    if params["mode"] == "train":
        model, hist = train(params)
    elif params["mode"] == "test":
        test_and_save(params)
    elif params["mode"] == "eval":
        evaluate(params)

    evals.test_accuracy(model, [mnistmx_test, mnistmx_test], mnisty_test_or)

    evals.test_accuracy(model, [mnistmx_test, mnistmx_test], mnisty_test_or)


if __name__ == "__main__":
    main()
