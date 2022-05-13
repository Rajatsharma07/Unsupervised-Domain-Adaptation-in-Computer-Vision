import tensorflow as tf
import os
import argparse
from src.train_test import train_test, evaluate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Below command selects the the particular GPU for training
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    # This command does performance improvements


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for the experiment")
    parser.add_argument(
        "--combination",
        type=str,
        default="Amazon_to_Webcam",
        help="Choose domain adaptation scenario, see config file.",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="Xception",
        help="Choose backbone models, see config file",
    )

    parser.add_argument(
        "--prune_val",
        type=float,
        default=0.30,
        help="Target sparsity value for pruning",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=299,
        help="Image resizing dimensions",
    )

    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=(299, 299, 3),
        help="Model's input shape",
    )

    parser.add_argument(
        "--output_classes",
        type=int,
        default=31,
        help="Classes in the dataset",
    )

    parser.add_argument("--batch_size", default=16, help="Batch size", type=int)

    parser.add_argument(
        "--learning_rate", default=0.001, help="Learning rate", type=float
    )

    parser.add_argument(
        "--mode",
        help="'train_test' or 'eval' options, see train_test.py module",
        default="train_test",
        type=str,
    )

    parser.add_argument(
        "--loss_function",
        help="Select domain alignment loss function, see loss.py module",
        default="CORAL",
        type=str,
    )

    parser.add_argument(
        "--lambda_loss",
        help="Weighting factor for domain alignment loss",
        default=0.50,
        type=float,
    )

    parser.add_argument(
        "--augment",  # Default set is false
        help="To apply data augmentation on the source dataset",
        action="store_true",
    )

    parser.add_argument(
        "--technique",  # Default set is false
        help="Choose techniques, MBM - if false, CDAN - if frue",
        action="store_true",
    )

    parser.add_argument(
        "--prune",  # Default set is false
        help="Shared FE will be optimized if MBM, otherwise Target FE will be optimized if CDAN",
        action="store_true",
    )

    parser.add_argument(
        "--epochs", default=4, help="Epochs to run a particular scenario", type=int
    )

    parser.add_argument(
        "--save_weights",  # Default set is false
        help="To save the intermediate weights of the model when improvement observed in the validation accuracy",
        action="store_true",
    )

    parser.add_argument(
        "--save_model",  # Default set is false
        help="If yes, final trained model will be saved, otherwise not",
        action="store_true",
    )

    parser.add_argument(
        "--use_multiGPU",  # Default set is false
        help="If yes, multiple single host GPUs will be used, otherwise not",
        action="store_true",
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
