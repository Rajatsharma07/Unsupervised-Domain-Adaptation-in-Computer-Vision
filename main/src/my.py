import pandas as pd
import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


train_dir = os.path.join(os.getcwd(), "data/mnistm/mnist_m_train")
df_train = pd.read_csv(
    train_dir + "/mnist_m_train_labels.txt",
    sep=" ",
    header=None,
)
file_paths_train = df_train[0].values
labels_train = df_train[1].values

test_dir = "data/mnistm/mnist_m_test"
df_test = pd.read_csv(
    test_dir + "/mnist_m_test_labels.txt",
    sep=" ",
    header=None,
)

file_paths_test = df_test[0].values[:5]
labels_test = df_test[1].values[:5]


ds_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
ds_test = tf.data.Dataset.from_tensor_slices((file_paths_test, labels_test))


mnist = tf.keras.datasets.mnist
(mnistx_train, mnisty_train), (mnistx_test, mnisty_test) = mnist.load_data()

ds_train_mnist = tf.data.Dataset.from_tensor_slices((mnistx_train, mnisty_train))
ds_test_mnist = tf.data.Dataset.from_tensor_slices((mnistx_test, mnisty_test))


def read_image(image_file, label, directory, is_greyscale):
    image = tf.io.read_file(directory + "/" + image_file)
    if is_greyscale:
        channels = 1
    else:
        channels = 3
    image = tf.image.decode_image(
        image, channels=channels, dtype=tf.float32, expand_animations=False
    )
    if is_greyscale:
        image = tf.tile(image, [1, 1, 3])
    return image, label


def resize_and_rescale(image, label, new_size, is_greyscale):
    image = tf.cast(image, tf.float32)
    if is_greyscale:
        image = tf.tile(image, [1, 1, 3])
    image = tf.image.resize(image, [new_size, new_size])
    image = image / 255.0
    return image, label


def augment(image, label, new_size, is_greyscale):
    image, label = resize_and_rescale(image, label, new_size, is_greyscale)
    # if int(channels) == 3:
    #     if tf.random.uniform((), minval=0, maxval=1) < 0.1:
    #         image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.5)
    image = tf.image.adjust_saturation(image, 3)

    # a left upside down flipped
    image = tf.image.random_flip_left_right(image)  # 50%

    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
# Read the data
# Setup for train dataset
ds_train = ds_train.map(
    (lambda x, y: read_image(x, y, train_dir, False)), num_parallel_calls=AUTOTUNE
)
ds_train = (
    ds_train.map((lambda x, y: augment(x, y, 32, False)), num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# def show(image, label):
#     plt.figure()
#     plt.imshow(image[0])
#     plt.title(str(label))
#     plt.axis("off")
#     plt.show()


count = 0
for image, label in ds_train:
    # show(image, label)
    count += 1
print(f"\n Count of MNISTM: {count}\n")


# (ds_train_mnist, ds_test), ds_info = tfds.load(
#     "mnist",
#     split=["train", "test"],
#     shuffle_files=True,
#     as_supervised=True,  # will return tuple (img, label) otherwise dict
#     with_info=True,  # able to get info about dataset
# )
# ds_train_mnist = ds_train_mnist.map(
#     (lambda x, y: augment(x, y, 32, True)), num_parallel_calls=AUTOTUNE
# )


# ds_train_mnist = (
#     # ds_train_mnist.shuffle(1000)
#     # .apply(tf.data.experimental.unbatch())
#     ds_train_mnist.batch(BATCH_SIZE).prefetch(AUTOTUNE)
# )
# count = 0
# for image, label in ds_train_mnist:
#     # show(image, label)
#     count += 1
# print(f"\n Count of MNIST Greyscale: {count}\n")
