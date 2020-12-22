import tensorflow as tf
import os
import argparse
from src import train_test as te

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for the experiment")
    parser.add_argument(
        "--combination",
        type=int,
        default=1,
        help="pass experiment combination, see config file",
    )

    parser.add_argument(
        "--model_mode",
        type=int,
        default="0",
        help="0 means SIMAESE network, otherwise seperate",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=32,
        help="pass image resizing dimension",
    )

    parser.add_argument("--batch_size", default=32, help="batch size", type=int)

    parser.add_argument(
        "--learning_rate", default=0.001, help="Learning rate", type=float
    )

    parser.add_argument(
        "--mode", help="train_test or eval options", default="train_test", type=str
    )

    parser.add_argument(
        "--lambda_loss", help="Additional loss lambda value", default=0.5, type=float
    )

    # parser.add_argument(
    #     "--source_data_dir", help="Source Data path", default="", type=str
    # )
    # parser.add_argument(
    #     "--target_data_dir", help="Target Data path", default="", type=str
    # )

    parser.add_argument("--epochs", default=50, help="Epochs", type=int)

    parser.add_argument(
        "--save_weights",
        default=False,
        help="If yes, weights will be saved, otherwise not",
        type=bool,
    )

    parser.add_argument(
        "--save_model",
        default=True,
        help="If yes, model will be saved, otherwise not",
        type=bool,
    )

    parser.add_argument(
        "--use_multiGPU",
        default=False,
        help="If yes, multiple single host GPUs will be used, otherwise not",
        type=bool,
    )

    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()
    params = vars(args)
    print(params)

    assert params[
        "mode"
    ], "mode is required, please provide either train_test or eval option"

    assert params["mode"] in [
        "train_test",
        "eval",
    ], "The mode must be train_test or eval"

    if params["mode"] == "train_test":
        model, hist, results = te.train_test(params)

    elif params["mode"] == "eval":
        pass


if __name__ == "__main__":
    main()
