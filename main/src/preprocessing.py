import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":
    import config as cn

    # from utils import extract_mnist_m
else:
    import src.config as cn

    # from src.utils import extract_mnist_m

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# amazon_ds = image_dataset_from_directory(
#     directory=cn.OFFICE_DS_PATH / "amazon",
#     labels="inferred",
#     label_mode="int",
#     batch_size=1,
#     image_size=(227, 227),
#     interpolation="nearest",
# )
# webcam_ds = image_dataset_from_directory(
#     directory=cn.OFFICE_DS_PATH / "webcam",
#     labels="inferred",
#     label_mode="int",
#     batch_size=1,
#     image_size=(227, 227),
#     interpolation="nearest",
# )
# dslr_ds = image_dataset_from_directory(
#     directory=cn.OFFICE_DS_PATH / "dslr",
#     labels="inferred",
#     label_mode="int",
#     batch_size=1,
#     image_size=(227, 227),
#     interpolation="nearest",
# )

# datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=10,
#     horizontal_flip=False,
#     vertical_flip=False,
#     fill_mode="nearest",
#     validation_split=0.0,
#     dtype=tf.float32,
#     preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
# )
# train_generator_amazon = datagen.flow_from_directory(
#     cn.OFFICE_DS_PATH / "amazon",
#     target_size=(227, 227),
#     batch_size=32,
#     color_mode="rgb",
#     class_mode="sparse",
#     shuffle=True,
#     subset="training",
# )


# train_generator_webcam = datagen.flow_from_directory(
#     cn.OFFICE_DS_PATH / "webcam",
#     target_size=(227, 227),
#     batch_size=32,
#     color_mode="rgb",
#     class_mode="sparse",
#     shuffle=True,
#     subset="training",
# )
# train_generator_dslr = datagen.flow_from_directory(
#     cn.OFFICE_DS_PATH / "dslr",
#     target_size=(227, 227),
#     batch_size=32,
#     color_mode="rgb",
#     class_mode="sparse",
#     shuffle=True,
#     subset="training",
# )


def generator_two_img():

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest",
        validation_split=0.0,
        dtype=tf.float32,
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    )
    # (img, label)
    train_generator_amazon = datagen.flow_from_directory(
        cn.OFFICE_DS_PATH / "amazon",
        target_size=(227, 227),
        batch_size=32,
        color_mode="rgb",
        class_mode="sparse",
        shuffle=True,
        subset="training",
        # seed=1,
    )

    # (img, label)
    train_generator_webcam = datagen.flow_from_directory(
        cn.OFFICE_DS_PATH / "webcam",
        target_size=(227, 227),
        batch_size=32,
        color_mode="rgb",
        class_mode="sparse",
        shuffle=True,
        subset="training",
        # seed=1,
    )
    # train_generator_dslr = datagen.flow_from_directory(
    #     os.path.join(train_path, "dslr"),
    #     target_size=(227, 227),
    #     batch_size=1,
    #     color_mode="rgb",
    #     class_mode="sparse",
    #     shuffle=True,
    #     subset="training",
    # )
    i = 0
    while i < 24:
        X1 = train_generator_amazon.__next__()
        X2 = train_generator_webcam.__next__()
        i += 1
        yield ((X1[0], X2[0]), X1[1])


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
    image = tf.image.resize(
        image, [new_size, new_size], antialias=True, method="nearest",
    )
    image = image / 255.0

    return image


def rescale_and_cast(src_image, tar_image, label):
    src_image = tf.cast(src_image, tf.float32) / 255.0
    src_image = tf.keras.applications.vgg16.preprocess_input(src_image)
    tar_image = tf.cast(tar_image, tf.float32) / 255.0
    tar_image = tf.keras.applications.vgg16.preprocess_input(tar_image)
    label = tf.cast(label, tf.uint8)
    return ((src_image, tar_image), label)


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
        # (mnistx_train, mnisty_train), (_, _) = tf.keras.datasets.mnist.load_data()

        # mnistmx_train, _ = extract_mnist_m(cn.MNIST_M_PATH)
        # mnistmx_train, mnistmy_train = shuffle_dataset(mnistmx_train, mnisty_train)
        # ds_train = tf.data.Dataset.from_tensor_slices(
        #     ((mnistx_train, mnistmx_train), mnisty_train)
        # )

        # ds_custom_val = tf.data.Dataset.from_tensor_slices(
        #     (mnistmx_train, mnistmy_train)
        # )

        # ds_test = tf.data.Dataset.from_tensor_slices(
        #     ((mnistmx_train, mnistmx_train), mnistmy_train)
        # )

        # # Read the data
        # # Setup for train dataset
        # ds_train = (
        #     ds_train.map(
        #         (lambda x, y: augment(x[0], x[1], y, params["resize"], True, False)),
        #         num_parallel_calls=cn.AUTOTUNE,
        #     )
        #     .cache()
        #     .batch(params["batch_size"])
        #     .prefetch(cn.AUTOTUNE)
        #     # .shuffle(mnisty_train.shape[0])
        # )

        # # Setup for test Dataset
        # ds_test = (
        #     ds_test.map(
        #         (
        #             lambda x, y: (
        #                 (
        #                     resize_and_rescale(x[0], params["resize"], False),
        #                     resize_and_rescale(x[1], params["resize"], False),
        #                 ),
        #                 y,
        #             )
        #         ),
        #         num_parallel_calls=cn.AUTOTUNE,
        #     )
        #     .batch(params["batch_size"])
        #     .prefetch(cn.AUTOTUNE)
        # )

        # # Setup for Custom Val Dataset
        # ds_custom_val = (
        #     ds_custom_val.map(
        #         (lambda x, y: (resize_and_rescale(x, params["resize"], False), y)),
        #         num_parallel_calls=cn.AUTOTUNE,
        #     )
        #     .batch(params["batch_size"])
        #     .prefetch(cn.AUTOTUNE)
        # )

        # source_count = [x for x in ds_train]
        # tf.compat.v1.logging.info(
        #     "Batch count of source training set: " + str(len(source_count))
        # )

        # test_count = [x for x in ds_test]
        # tf.compat.v1.logging.info("Batch count of test set: " + str(len(test_count)))

        # val_count = [x for x in ds_custom_val]
        # tf.compat.v1.logging.info("Batch count of val set: " + str(len(val_count)))

        # return ds_train, ds_custom_val, ds_test
        pass

    elif params["combination"] == 2:

        # (mnistx_train, mnisty_train), (_, _) = tf.keras.datasets.mnist.load_data()

        # mnistmx_train, _ = extract_mnist_m(cn.MNIST_M_PATH)
        # mnistmy_train = mnisty_train

        # mnistx_train, mnisty_train = shuffle_dataset(mnistx_train, mnisty_train)

        # ds_train = tf.data.Dataset.from_tensor_slices(
        #     ((mnistmx_train, mnistx_train), mnistmy_train)
        # )

        # ds_test = tf.data.Dataset.from_tensor_slices(
        #     ((mnistx_train, mnistx_train), mnisty_train)
        # )

        # # Read the data
        # # Setup for train dataset
        # ds_train = (
        #     ds_train.map(
        #         (lambda x, y: augment(x[0], x[1], y, params["resize"], False, True)),
        #         num_parallel_calls=cn.AUTOTUNE,
        #     )
        #     .cache()
        #     .batch(params["batch_size"])
        #     .prefetch(cn.AUTOTUNE)
        #     # .shuffle(mnisty_train.shape[0])
        # )

        # # Setup for test Dataset
        # ds_test = (
        #     ds_test.map(
        #         (
        #             lambda x, y: (
        #                 (
        #                     resize_and_rescale(x[0], params["resize"], True),
        #                     resize_and_rescale(x[1], params["resize"], True),
        #                 ),
        #                 y,
        #             )
        #         ),
        #         num_parallel_calls=cn.AUTOTUNE,
        #     )
        #     .batch(params["batch_size"])
        #     .prefetch(cn.AUTOTUNE)
        # )

        # source_count = [x for x in ds_train]
        # tf.compat.v1.logging.info(
        #     "Batch count of training set: " + str(len(source_count))
        # )

        # test_count = [x for x in ds_test]
        # tf.compat.v1.logging.info("Batch count of test set: " + str(len(test_count)))

        # return ds_train, _, ds_test
        pass

    elif params["combination"] == 3:
        # source = []
        # target = []
        # source_labels = []
        # target_labels = []

        for x, y in tf.data.Dataset.zip((amazon_ds, webcam_ds.repeat(126))):
            source.append(tf.squeeze(x[0], axis=0))
            target.append(tf.squeeze(y[0], axis=0))
            source_labels.append(x[1])
            target_labels.append(y[1])

        # ds_train = (
        #     tf.data.Dataset.from_tensor_slices(((source, target), source_labels))
        #     .map(
        #         lambda x, y: rescale_and_cast(x[0], x[1], y),
        #         num_parallel_calls=cn.AUTOTUNE,
        #     )
        #     .shuffle(len(source))
        #     .batch(params["batch_size"])
        #     .prefetch(cn.AUTOTUNE)
        # )
        # ds_test = (
        #     tf.data.Dataset.from_tensor_slices(((target, target), target_labels))
        #     .map(
        #         lambda x, y: rescale_and_cast(x[0], x[1], y),
        #         num_parallel_calls=cn.AUTOTUNE,
        #     )
        #     .shuffle(len(target))
        #     .batch(params["batch_size"])
        #     .prefetch(cn.AUTOTUNE)
        # )
        # return ds_train, ds_test, ds_test
        ds_train = generator_two_img()

        return ds_train, ds_train, ds_train


if __name__ == "__main__":
    dsds = generator_two_img()
    count = 0
    for x in dsds:
        count += 1
    print("Hello")
