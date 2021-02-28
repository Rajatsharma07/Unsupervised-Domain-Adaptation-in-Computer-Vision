import os
import tensorflow as tf
import src.config as cn
import math
import pandas as pd
from src.utils import extract_mnist_m, shuffle_dataset
import tensorflow_datasets as tfds
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def augment_ds(image, label, prob=0.2):

    # Make Images Greyscale
    image = tf.cond(
        tf.random.uniform(shape=[], minval=0, maxval=1) < prob,
        lambda: tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 1, 3]),
        lambda: image,
    )

    # Adding Gaussian Noise
    noise = tf.random.normal(
        shape=tf.shape(image), mean=0.0, stddev=1, dtype=tf.float32
    )
    image = tf.cond(
        tf.random.uniform(shape=[], minval=0, maxval=1) < prob,
        lambda: tf.add(image, noise),
        lambda: image,
    )

    # Colour Augmentations
    image = tf.image.random_hue(image, 0.05)
    image = tf.image.random_saturation(image, 0.8, 2.5)
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.2)

    # Rotating Images
    image = tf.cond(
        tf.random.uniform(shape=[], minval=0, maxval=1) < prob,
        lambda: tf.image.rot90(image, k=1),
        lambda: image,
    )

    # Flipping Images
    image = tf.image.random_flip_left_right(image)

    return image, label


def read_from_file(image_file, label):
    directory = "/root/Master-Thesis/code/data/synthetic_data/train/"
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [71, 71], method="nearest")
    image = tf.keras.applications.xception.preprocess_input(image)
    label = tf.cast(label, tf.float32)

    return image, label


def read_images(directory, batch_size, new_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",  # categorical, binary
        batch_size=batch_size,
        image_size=(new_size, new_size),  # reshape if not in this size
        shuffle=True,
    )
    return ds


def preprocess(image, label):
    # Cast to float32
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    image = tf.keras.applications.xception.preprocess_input(image)
    # image = image / 255.0

    return image, label


def prepare_office_ds(source_directory, target_directory, params):

    source_ds_original = read_images(
        source_directory, params["batch_size"], params["resize"]
    )
    target_ds_original = read_images(
        target_directory, params["batch_size"], params["resize"]
    )

    source_ds_original = source_ds_original.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    target_ds_original = target_ds_original.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    length_source_images = source_ds_original.cardinality().numpy()
    tf.compat.v1.logging.info(f"length_source_images: {length_source_images}")
    length_target_images = target_ds_original.cardinality().numpy()
    tf.compat.v1.logging.info(f"length_target_images: {length_target_images}")

    if length_source_images < length_target_images:
        source_ds = source_ds_original.repeat(
            math.ceil(length_target_images / length_source_images)
        )
        target_ds = target_ds_original
    elif length_source_images > length_target_images:
        target_ds = target_ds_original.repeat(
            math.ceil(length_source_images / length_target_images)
        )
        source_ds = source_ds_original

    else:
        source_ds = source_ds_original
        target_ds = target_ds_original

    if params["augment"]:
        source_ds = source_ds.map(augment_ds, num_parallel_calls=cn.AUTOTUNE)

    ds_train = tf.data.Dataset.zip((source_ds, target_ds)).map(
        lambda x, y: ((x[0], y[0]), x[1]), num_parallel_calls=cn.AUTOTUNE
    )
    ds_train = ds_train.prefetch(buffer_size=cn.AUTOTUNE)

    ds_test = target_ds_original.map(lambda x, y: ((x, x), y)).prefetch(
        buffer_size=cn.AUTOTUNE
    )

    train_count = [x for x in ds_train]
    tf.compat.v1.logging.info("Batch count of training set: " + str(len(train_count)))

    test_count = [x for x in ds_test]
    tf.compat.v1.logging.info("Batch count of test set: " + str(len(test_count)))

    return ds_train, ds_test


def fetch_data(params):

    if cn.DATASET_COMBINATION[params["combination"]] == 1:
        source_directory = cn.OFFICE_DS_PATH / "amazon"
        target_directory = cn.OFFICE_DS_PATH / "webcam"

        return prepare_office_ds(source_directory, target_directory, params)

    elif cn.DATASET_COMBINATION[params["combination"]] == 2:
        source_directory = cn.OFFICE_DS_PATH / "amazon"
        target_directory = cn.OFFICE_DS_PATH / "dslr"

        return prepare_office_ds(source_directory, target_directory, params)

    elif cn.DATASET_COMBINATION[params["combination"]] == 3:
        source_directory = cn.OFFICE_DS_PATH / "webcam"
        target_directory = cn.OFFICE_DS_PATH / "amazon"

        return prepare_office_ds(source_directory, target_directory, params)

    elif cn.DATASET_COMBINATION[params["combination"]] == 4:
        source_directory = cn.OFFICE_DS_PATH / "dslr"
        target_directory = cn.OFFICE_DS_PATH / "amazon"

        return prepare_office_ds(source_directory, target_directory, params)

    elif cn.DATASET_COMBINATION[params["combination"]] == 5:

        synthetic_directory = cn.SYNTHETIC_PATH

        labels_data = pd.read_csv(
            synthetic_directory / "train_labelling.txt", sep=" ", header=None
        )

        file_paths = labels_data[0].str[6:]
        labels = labels_data[1].values

        ds_source = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        ds_source = ds_source.map(read_from_file, num_parallel_calls=cn.AUTOTUNE).batch(
            params["batch_size"]
        )

        GTSRB_directory = cn.GTSRB_PATH
        target_ds_original = read_images(
            GTSRB_directory / Path("train"), params["batch_size"], params["resize"]
        )

        target_ds_original = target_ds_original.map(
            preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        length_source_images = ds_source.cardinality().numpy()
        tf.compat.v1.logging.info(f"length_source_images: {length_source_images}")
        length_target_images = target_ds_original.cardinality().numpy()
        tf.compat.v1.logging.info(f"length_target_images: {length_target_images}")

        ds_target = target_ds_original.repeat(
            math.ceil(length_source_images / length_target_images)
        )

        if params["augment"]:
            ds_source = ds_source.map(augment_ds, num_parallel_calls=cn.AUTOTUNE)

        ds_train = tf.data.Dataset.zip((ds_source, ds_target)).map(
            lambda x, y: ((x[0], y[0]), x[1]), num_parallel_calls=cn.AUTOTUNE
        )
        ds_train = ds_train.prefetch(buffer_size=cn.AUTOTUNE)

        ds_test = target_ds_original.map(lambda x, y: ((x, x), y)).prefetch(
            buffer_size=cn.AUTOTUNE
        )

        train_count = [x for x in ds_train]
        tf.compat.v1.logging.info(
            "Batch count of training set: " + str(len(train_count))
        )

        test_count = [x for x in ds_test]
        tf.compat.v1.logging.info("Batch count of test set: " + str(len(test_count)))

        return ds_train, ds_test
