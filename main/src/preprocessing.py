import os
import tensorflow as tf
import numpy as np
import src.config as cn
from src.utils import extract_mnist_m

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def shuffle_dataset(data_x, data_Y):
    # np.random.seed(seed_val)
    index_shuffled = np.arange(data_x.shape[0])
    np.random.shuffle(index_shuffled)
    data_x = np.array(data_x)
    data_x = data_x[index_shuffled]
    data_Y = (np.array(data_Y))[index_shuffled]
    return data_x, data_Y


def resize_and_rescale(image, new_size, is_greyscale):
    image = tf.cast(image, tf.float32)
    if is_greyscale:
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)
        # image = tf.tile(image, [1, 1, 3])
    image = tf.image.resize(
        image,
        [new_size, new_size],
        antialias=True,
        method="nearest",
    )
    image = image / 255.0

    return image


def augment(
    image_source,
    image_target,
    label,
    new_size,
    source_is_greyscale,
    target_is_greyscale,
):
    image0 = resize_and_rescale(image_source, new_size, source_is_greyscale)
    image1 = resize_and_rescale(image_target, new_size, target_is_greyscale)
    # if int(channels) == 3:
    #     if tf.random.uniform((), minval=0, maxval=1) < 0.1:
    #         image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])
    image0 = tf.image.random_brightness(image0, max_delta=0.5)
    image0 = tf.image.random_contrast(image0, lower=0.1, upper=0.5)
    image0 = tf.image.adjust_saturation(image0, 3)
    # a left upside down flipped
    image0 = tf.image.random_flip_left_right(image0)  # 50%

    return ((image0, image1), label)


def fetch_data(params):
    if (params["combination"]) == 1:
        (mnistx_train, mnisty_train), (_, _) = tf.keras.datasets.mnist.load_data()

        mnistmx_train, _ = extract_mnist_m(cn.MNIST_M_PATH)
        mnistmx_train, mnistmy_train = shuffle_dataset(mnistmx_train, mnisty_train)
        ds_train = tf.data.Dataset.from_tensor_slices(
            ((mnistx_train, mnistmx_train), mnisty_train)
        )

        ds_custom_val = tf.data.Dataset.from_tensor_slices(
            (mnistmx_train, mnistmy_train)
        )

        ds_test = tf.data.Dataset.from_tensor_slices(
            ((mnistmx_train, mnistmx_train), mnistmy_train)
        )

        # Read the data
        # Setup for train dataset
        ds_train = (
            ds_train.map(
                (lambda x, y: augment(x[0], x[1], y, params["resize"], True, False)),
                num_parallel_calls=cn.AUTOTUNE,
            )
            .cache()
            .batch(params["batch_size"])
            .prefetch(cn.AUTOTUNE)
            # .shuffle(mnisty_train.shape[0])
        )

        # Setup for test Dataset
        ds_test = (
            ds_test.map(
                (
                    lambda x, y: (
                        (
                            resize_and_rescale(x[0], params["resize"], False),
                            resize_and_rescale(x[1], params["resize"], False),
                        ),
                        y,
                    )
                ),
                num_parallel_calls=cn.AUTOTUNE,
            )
            .batch(params["batch_size"])
            .prefetch(cn.AUTOTUNE)
        )

        # Setup for Custom Val Dataset
        ds_custom_val = (
            ds_custom_val.map(
                (lambda x, y: (resize_and_rescale(x, params["resize"], False), y)),
                num_parallel_calls=cn.AUTOTUNE,
            )
            .batch(params["batch_size"])
            .prefetch(cn.AUTOTUNE)
        )

        source_count = [x for x in ds_train]
        tf.compat.v1.logging.info(
            "Batch count of source training set: " + str(len(source_count))
        )

        test_count = [x for x in ds_test]
        tf.compat.v1.logging.info("Batch count of test set: " + str(len(test_count)))

        val_count = [x for x in ds_custom_val]
        tf.compat.v1.logging.info("Batch count of val set: " + str(len(val_count)))

        return ds_train, ds_custom_val, ds_test

    elif params["combination"] == 2:

        (mnistx_train, mnisty_train), (_, _) = tf.keras.datasets.mnist.load_data()

        mnistmx_train, _ = extract_mnist_m("/root/Master-Thesis/data/keras_mnistm.pkl")
        mnistmy_train = mnisty_train

        mnistx_train, mnisty_train = shuffle_dataset(mnistx_train, mnisty_train)

        ds_train = tf.data.Dataset.from_tensor_slices(
            ((mnistmx_train, mnistx_train), mnistmy_train)
        )

        ds_test = tf.data.Dataset.from_tensor_slices(
            ((mnistx_train, mnistx_train), mnisty_train)
        )

        # Read the data
        # Setup for train dataset
        ds_train = (
            ds_train.map(
                (lambda x, y: augment(x[0], x[1], y, params["resize"], False, True)),
                num_parallel_calls=cn.AUTOTUNE,
            )
            .cache()
            .batch(params["batch_size"])
            .prefetch(cn.AUTOTUNE)
            # .shuffle(mnisty_train.shape[0])
        )

        # Setup for test Dataset
        ds_test = (
            ds_test.map(
                (
                    lambda x, y: (
                        (
                            resize_and_rescale(x[0], params["resize"], True),
                            resize_and_rescale(x[1], params["resize"], True),
                        ),
                        y,
                    )
                ),
                num_parallel_calls=cn.AUTOTUNE,
            )
            .batch(params["batch_size"])
            .prefetch(cn.AUTOTUNE)
        )

        source_count = [x for x in ds_train]
        tf.compat.v1.logging.info(
            "Batch count of training set: " + str(len(source_count))
        )

        test_count = [x for x in ds_test]
        tf.compat.v1.logging.info("Batch count of test set: " + str(len(test_count)))

        return ds_train, _, ds_test

    elif params["combination"] == 3:
        pass
