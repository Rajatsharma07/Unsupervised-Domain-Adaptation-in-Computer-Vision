import pickle
import tensorflow as tf
import numpy as np
import pandas as pd

# train_dir = "data/mnist_images_csv/"
# df_train = pd.read_csv(train_dir + "train.csv")

# test_dir = "data/mnist_images_csv/"
# df_test = pd.read_csv(test_dir + "test.csv")

# file_paths_train = df_train["file_name"].values
# labels_train = df_train["label"].values

# file_paths_test = df_test["file_name"].values
# labels_test = df_test["label"].values

# ds_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
# ds_test = tf.data.Dataset.from_tensor_slices((file_paths_test, labels_test))


def read_image(image_file, label, directory, channels=3):
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_image(image, channels=channels, dtype=tf.float32)
    return image, label


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


def augment(image, label, size=32):
    new_height = new_width = size
    image = tf.image.resize(image, (new_height, new_width))

    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2)

    # a left upside down flipped is still a dog ;)
    image = tf.image.random_flip_left_right(image)  # 50%
    # image = tf.image.random_flip_up_down(image) #%50%

    return image, label


# AUTOTUNE = tf.data.experimental.AUTOTUNE
# BATCH_SIZE = cn.BATCH_SIZE

# # Read the data
# ds_train = ds_train.map(
#     read_image(directory=train_dir, channels=1), num_parallel_calls=AUTOTUNE
# )

# # Setup for train dataset
# ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
# ds_train = ds_train.cache()
# # ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
# ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
# ds_train = ds_train.batch(BATCH_SIZE)
# ds_train = ds_train.prefetch(AUTOTUNE)

# # Setup for test Dataset
# ds_test = ds_train.map(
#     read_image(directory=test_dir, channels=3), num_parallel_calls=AUTOTUNE
# )
# ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
# ds_test = ds_train.batch(BATCH_SIZE)
# ds_test = ds_train.prefetch(AUTOTUNE)


def scale_normalize(dataset_train, dataset_test):
    dataset_train_normalized = tf.cast(dataset_train, tf.float32)
    dataset_train_normalized = tf.keras.applications.resnet50.preprocess_input(
        dataset_train_normalized
    )

    dataset_test_normalized = tf.cast(dataset_test, tf.float32)
    dataset_test_normalized = tf.keras.applications.resnet50.preprocess_input(
        dataset_test_normalized
    )
    return dataset_train_normalized, dataset_test_normalized


def get_mnist(preTrained=True):
    mnist = tf.keras.datasets.mnist
    (mnistx_train, mnisty_train), (mnistx_test, mnisty_test) = mnist.load_data()

    print(
        f"mnistx_train shape: {mnistx_train.shape}, mnisty_train shape : {mnisty_train.shape}"
    )
    print(
        f"mnistx_test shape: {mnisty_test.shape}, mnisty_test shape : {mnisty_test.shape}"
    )

    # Preprocessing
    mnistx_train = tf.image.resize(
        tf.reshape(mnistx_train, shape=[-1, 28, 28, 1]),
        [32, 32],
        method="nearest",
        preserve_aspect_ratio=False,
        antialias=True,
        name=None,
    )
    mnistx_test = tf.image.resize(
        tf.reshape(mnistx_test, shape=[-1, 28, 28, 1]),
        [32, 32],
        method="nearest",
        preserve_aspect_ratio=False,
        antialias=True,
        name=None,
    )

    mnistx_train = tf.image.grayscale_to_rgb(mnistx_train)
    mnistx_test = tf.image.grayscale_to_rgb(mnistx_test)

    if preTrained:
        mnistx_train, mnistx_test = scale_normalize(mnistx_train, mnistx_test)
    else:
        mnistx_train = mnistx_train.astype("float32") / 255.0
        mnistx_test = mnistx_test.astype("float32") / 255.0

    return mnistx_train, mnisty_train, mnistx_test, mnisty_test


def get_mnist_m(mnistm_path, preTrained=True):
    with open(mnistm_path, "rb") as f:
        mnistm_dataset = pickle.load(f, encoding="bytes")

    mnistmx_train = mnistm_dataset[b"train"]
    mnistmx_test = mnistm_dataset[b"test"]
    print(mnistmx_train.shape, mnistmx_test.shape)

    # Preprocessing for Keras inbuilt models - ResNet 50
    # Resizing (minimum 32*32)
    mnistmx_train = tf.image.resize(
        mnistmx_train,
        [32, 32],
        method="nearest",
        preserve_aspect_ratio=False,
        antialias=True,
        name=None,
    )
    mnistmx_test = tf.image.resize(
        mnistmx_test,
        [32, 32],
        method="nearest",
        preserve_aspect_ratio=False,
        antialias=True,
        name=None,
    )

    if preTrained:
        mnistmx_train, mnistmx_test = scale_normalize(mnistmx_train, mnistmx_test)
    else:
        mnistmx_train = mnistmx_train.astype("float32") / 255.0
        mnistmx_test = mnistmx_test.astype("float32") / 255.0

    # shuffle the dataset

    return mnistmx_train, mnistmx_test


def shuffle_dataset(data_x, data_Y, seed_val):
    np.random.seed(seed_val)
    index_shuffled = np.arange(data_x.shape[0])
    np.random.shuffle(index_shuffled)

    data_x = np.array(data_x)

    data_x = data_x[index_shuffled]
    data_Y = (np.array(data_Y))[index_shuffled]

    return data_x, data_Y
