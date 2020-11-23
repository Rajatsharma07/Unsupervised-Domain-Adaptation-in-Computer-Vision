import os
import tensorflow as tf
import numpy as np
import config as cn
from utils import extract_mnist_m

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def shuffle_dataset(data_x, data_Y, seed_val):
    np.random.seed(seed_val)
    index_shuffled = np.arange(data_x.shape[0])
    np.random.shuffle(index_shuffled)
    data_x = np.array(data_x)
    data_x = data_x[index_shuffled]
    data_Y = (np.array(data_Y))[index_shuffled]
    return data_x, data_Y


def resize_and_rescale(image, label, new_size, is_greyscale, is_pretrained=False):
    image = tf.cast(image, tf.float32)
    if is_greyscale:
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)
        # image = tf.tile(image, [1, 1, 3])
    image = tf.image.resize(image, [new_size, new_size])
    image = image / 255.0

    return image, label


def augment(
    image_source,
    image_target,
    label,
    new_size,
    source_is_greyscale,
    target_is_greyscale,
):
    image0, label = resize_and_rescale(
        image_source, label, new_size, source_is_greyscale
    )
    image1, _ = resize_and_rescale(image_target, label, new_size, target_is_greyscale)
    # if int(channels) == 3:
    #     if tf.random.uniform((), minval=0, maxval=1) < 0.1:
    #         image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])
    image0 = tf.image.random_brightness(image0, max_delta=0.5)
    image0 = tf.image.random_contrast(image0, lower=0.1, upper=0.5)
    image0 = tf.image.adjust_saturation(image0, 3)
    # a left upside down flipped
    image0 = tf.image.random_flip_left_right(image0)  # 50%

    return (image0, image1, label)


def fetch_data(params):
    if (params["combination"]) == 1:
        (mnistx_train, mnisty_train), (_, _) = tf.keras.datasets.mnist.load_data()

        mnistmx_train, _ = extract_mnist_m("/root/Master-Thesis/data/keras_mnistm.pkl")
        mnistmx_train, mnistmy_train = shuffle_dataset(
            mnistmx_train, mnisty_train, params["sample_seed"]
        )
        ds_train = tf.data.Dataset.from_tensor_slices(
            (mnistx_train, mnistmx_train, mnisty_train)
        )

        ds_test = tf.data.Dataset.from_tensor_slices((mnistmx_train, mnistmy_train))

        # Read the data
        # Setup for train dataset
        ds_train = (
            ds_train.map(
                (lambda x, y, z: augment(x, y, z, params["resize"], True, False)),
                num_parallel_calls=cn.AUTOTUNE,
            )
            .cache()
            .shuffle(1000)
            .batch(params["batch_size"])
            .prefetch(cn.AUTOTUNE)
        )
        # Setup for test Dataset
        ds_test = (
            ds_test.map(
                (lambda x, y: resize_and_rescale(x, y, params["resize"], False)),
                num_parallel_calls=cn.AUTOTUNE,
            )
            .batch(params["batch_size"])
            .prefetch(cn.AUTOTUNE)
        )
        count = 0
        for x in ds_train:
            count += 1
        print(f"\n Batches count of MNIST train: {count}\n")

        count = 0
        for x in ds_test:
            count += 1
        print(f"\n Batches Count of MNISTM test: {count}\n")

        return ds_train, ds_test

    elif params["combination"] == 2:

        (mnistx_train, mnisty_train), (_, _) = tf.keras.datasets.mnist.load_data()

        mnistmx_train, _ = extract_mnist_m("/root/Master-Thesis/data/keras_mnistm.pkl")
        mnistmy_train = mnisty_train

        mnistx_train, mnisty_train = shuffle_dataset(
            mnistx_train, mnisty_train, params["sample_seed"]
        )

        ds_train = tf.data.Dataset.from_tensor_slices(
            (mnistmx_train, mnistx_train, mnistmy_train)
        )

        ds_test = tf.data.Dataset.from_tensor_slices((mnistx_train, mnisty_train))

        # Read the data
        # Setup for train dataset
        ds_train = (
            ds_train.map(
                (lambda x, y, z: augment(x, y, z, params["resize"], False, True)),
                num_parallel_calls=cn.AUTOTUNE,
            )
            .cache()
            .shuffle(1000)
            .batch(params["batch_size"])
            .prefetch(cn.AUTOTUNE)
        )
        # Setup for test Dataset
        ds_test = (
            ds_test.map(
                (lambda x, y: resize_and_rescale(x, y, params["resize"], True)),
                num_parallel_calls=cn.AUTOTUNE,
            )
            .batch(params["batch_size"])
            .prefetch(cn.AUTOTUNE)
        )
        count = 0
        for x in ds_train:
            count += 1
        print(f"\n Batches count of MNISTM train: {count}\n")

        count = 0
        for x in ds_test:
            count += 1
        print(f"\n Batches Count of MNIST test: {count}\n")

        return ds_train, ds_test
    
    elif params["combination"] == 3:
        pass