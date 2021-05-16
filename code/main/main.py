import tensorflow as tf
import os
import argparse
from src.train_test import train_test, evaluate
import numpy as np

# from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for the experiment")
    parser.add_argument(
        "--combination",
        type=str,
        default="Amazon_to_Webcam",
        help="pass experiment combination, see config file.",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="Xception",
        help="Xception, see config file",
    )

    parser.add_argument(
        "--prune_val",
        type=float,
        default=0.30,
        help="percentage of Pruning",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=299,
        help="pass image resizing dimension",
    )

    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=(299, 299, 3),
        help="model input shape",
    )

    parser.add_argument(
        "--output_classes",
        type=int,
        default=31,
        help="classes in the dataset",
    )

    parser.add_argument("--batch_size", default=16, help="batch size", type=int)

    parser.add_argument(
        "--learning_rate", default=0.001, help="Learning rate", type=float
    )

    parser.add_argument(
        "--mode", help="train_test or eval options", default="eval", type=str
    )

    parser.add_argument(
        "--loss_function",
        help="CORAL/Another",
        default="CORAL",
        type=str,
    )

    parser.add_argument(
        "--lambda_loss", help="Additional loss lambda value", default=0.50, type=float
    )

    parser.add_argument(
        "--augment",
        help="Augmentation will be applied or not",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--technique",
        help="Base technique - if False, Ours - if True",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--prune",
        help="Model will be optimized in Original technique, otherwise Target model in our technique",
        default=False,
        type=bool,
    )

    parser.add_argument("--epochs", default=4, help="Epochs", type=int)

    parser.add_argument(
        "--save_weights",
        default=True,
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
        model, hist, results = train_test(params)

    elif params["mode"] == "eval":
        evaluate(
            model_path="/root/Master-Thesis/code/model_data/1_Xception_CORAL_0.5_Original/20210306-172240/model",
            params=params,
        )


if __name__ == "__main__":
    main()
