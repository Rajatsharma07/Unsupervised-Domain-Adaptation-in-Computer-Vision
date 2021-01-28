import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Dense, concatenate, ReLU, Lambda
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

initializer = tf.keras.initializers.he_normal()
BASE_DIR = Path()  # Base path
OFFICE_DS_PATH = BASE_DIR / Path("data/office/")
weight_path_alexnet = BASE_DIR / Path("model_data/pretrained_weights")
source_directory = OFFICE_DS_PATH / "amazon"

img_height = 227
img_width = 227
batch_size = 16

model = keras.Sequential(
    [
        layers.Input((227, 227, 3)),
        layers.Conv2D(128, 3, padding="same"),
        layers.Conv2D(64, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(31),
    ]
)


def AlexNet(
    img_shape=(227, 227, 3),
    num_classes=31,
    weights=weight_path_alexnet / "bvlc_alexnet.npy",
):
    input = tf.keras.Input(img_shape)

    conv1 = conv2d_bn(
        x=input,
        filters=96,
        kernel_size=11,
        strides=4,
        pad="SAME",
        group=1,
        name="conv1",
    )
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=3, strides=2)(conv1)
    conv2 = conv2d_bn(
        x=pool1,
        filters=256,
        kernel_size=5,
        strides=1,
        pad="SAME",
        group=2,
        name="conv2",
    )
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=3, strides=2)(conv2)
    conv3 = conv2d_bn(
        x=pool2,
        filters=384,
        kernel_size=3,
        strides=1,
        pad="SAME",
        activation="relu",
        group=1,
        name="conv3",
    )
    conv4 = conv2d_bn(
        x=conv3,
        filters=384,
        kernel_size=3,
        strides=1,
        pad="SAME",
        activation="relu",
        group=2,
        name="conv4",
    )
    conv5 = conv2d_bn(
        x=conv4,
        filters=256,
        kernel_size=3,
        strides=1,
        pad="SAME",
        activation="relu",
        group=2,
        name="conv5",
    )
    pool5 = layers.MaxPooling2D(pool_size=3, strides=2)(conv5)
    flatten5 = layers.Flatten()(pool5)
    fc6 = layers.Dense(4096, activation="relu", name="fc6")(flatten5)
    drop6 = layers.Dropout(0.5)(fc6)
    fc7 = layers.Dense(4096, activation="relu", name="fc7")(drop6)
    drop7 = layers.Dropout(0.5)(fc7)
    fc8 = layers.Dense(
        num_classes,
        name="fc8",
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
    )(drop7)
    model = keras.models.Model(input, fc8)
    print("AlexNet created.")

    if weights is not None:
        weights_dic = np.load(weights, encoding="bytes", allow_pickle=True).item()
        # model.set_weights(weights_dic)
        conv1w = weights_dic["conv1"][0]
        conv1b = weights_dic["conv1"][1]
        model.get_layer("conv1").set_weights([conv1w, conv1b])

        conv2w = weights_dic["conv2"][0]
        conv2b = weights_dic["conv2"][1]
        w_a, w_b = np.split(conv2w, 2, axis=-1)
        b_a, b_b = np.split(conv2b, 2, axis=-1)
        model.get_layer("conv2a").set_weights([w_a, b_a])
        model.get_layer("conv2b").set_weights([w_b, b_b])

        conv3w = weights_dic["conv3"][0]
        conv3b = weights_dic["conv3"][1]
        model.get_layer("conv3").set_weights([conv3w, conv3b])

        conv4w = weights_dic["conv4"][0]
        conv4b = weights_dic["conv4"][1]
        w_a, w_b = np.split(conv4w, 2, axis=-1)
        b_a, b_b = np.split(conv4b, 2, axis=-1)
        model.get_layer("conv4a").set_weights([w_a, b_a])
        model.get_layer("conv4b").set_weights([w_b, b_b])

        conv5w = weights_dic["conv5"][0]
        conv5b = weights_dic["conv5"][1]
        w_a, w_b = np.split(conv5w, 2, axis=-1)
        b_a, b_b = np.split(conv5b, 2, axis=-1)
        model.get_layer("conv5a").set_weights([w_a, b_a])
        model.get_layer("conv5b").set_weights([w_b, b_b])

        fc6w = weights_dic["fc6"][0]
        fc6b = weights_dic["fc6"][1]
        model.get_layer("fc6").set_weights([fc6w, fc6b])

        fc7w = weights_dic["fc7"][0]
        fc7b = weights_dic["fc7"][1]
        model.get_layer("fc7").set_weights([fc7w, fc7b])

        # fc8w = weights_dic["fc8"][0]
        # fc8b = weights_dic["fc8"][1]
        # model.get_layer("fc8").set_weights([fc8w, fc8b])

        print("Weights loaded.")

    return model


def conv2d_bn(
    x, filters, kernel_size, strides, pad, name, activation="linear", group=1
):
    # group = 1 or 2
    if group == 1:
        x = layers.Conv2D(
            filters,
            kernel_size,
            padding=pad,
            strides=strides,
            activation=activation,
            name=name,
        )(x)
    else:
        x_a, x_b = layers.Lambda(lambda x: tf.split(x, group, axis=-1))(x)
        x_a = layers.Conv2D(
            filters // 2,
            kernel_size,
            padding=pad,
            strides=strides,
            activation=activation,
            name=name + "a",
        )(x_a)
        x_b = layers.Conv2D(
            filters // 2,
            kernel_size,
            padding=pad,
            strides=strides,
            activation=activation,
            name=name + "b",
        )(x_b)
        x = layers.concatenate([x_a, x_b])
    return x


# def AlexNet(
#     img_shape=(227, 227, 3),
#     num_classes=31,
#     weights=weight_path_alexnet / "bvlc_alexnet.npy",
# ):
#     input = Input(img_shape)

#     conv1 = conv2d_bn(
#         x=input,
#         filters=96,
#         kernel_size=11,
#         strides=4,
#         pad="SAME",
#         group=1,
#         name="conv1",
#     )
#     conv1 = BatchNormalization()(conv1)
#     conv1 = ReLU()(conv1)
#     pool1 = MaxPooling2D(pool_size=3, strides=2)(conv1)
#     conv2 = conv2d_bn(
#         x=pool1,
#         filters=256,
#         kernel_size=5,
#         strides=1,
#         pad="SAME",
#         group=2,
#         name="conv2",
#     )
#     conv2 = ReLU()(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling2D(pool_size=3, strides=2)(conv2)
#     conv3 = conv2d_bn(
#         x=pool2,
#         filters=384,
#         kernel_size=3,
#         strides=1,
#         pad="SAME",
#         activation="relu",
#         group=1,
#         name="conv3",
#     )
#     conv4 = conv2d_bn(
#         x=conv3,
#         filters=384,
#         kernel_size=3,
#         strides=1,
#         pad="SAME",
#         activation="relu",
#         group=2,
#         name="conv4",
#     )
#     conv5 = conv2d_bn(
#         x=conv4,
#         filters=256,
#         kernel_size=3,
#         strides=1,
#         pad="SAME",
#         activation="relu",
#         group=2,
#         name="conv5",
#     )
#     pool5 = MaxPooling2D(pool_size=3, strides=2)(conv5)
#     flatten5 = Flatten()(pool5)
#     fc6 = Dense(4096, activation="relu", name="fc6")(flatten5)
#     fc6 = Dropout(0.5)(fc6)
#     fc7 = Dense(4096, activation="relu", name="fc7")(fc6)
#     fc7 = Dropout(0.5)(fc7)
#     fc8 = Dense(
#         num_classes,
#         # activation="softmax",
#         kernel_initializer=tf.initializers.RandomNormal(0, 0.005),
#         name="fc8",
#     )(fc7)

#     model = Model(input, fc8)
#     print("AlexNet created.")

#     if weights is not None:
#         weights_dic = np.load(weights, encoding="bytes", allow_pickle=True).item()
#         # model.set_weights(weights_dic)
#         conv1w = weights_dic["conv1"][0]
#         conv1b = weights_dic["conv1"][1]
#         model.get_layer("conv1").set_weights([conv1w, conv1b])

#         conv2w = weights_dic["conv2"][0]
#         conv2b = weights_dic["conv2"][1]
#         w_a, w_b = np.split(conv2w, 2, axis=-1)
#         b_a, b_b = np.split(conv2b, 2, axis=-1)
#         model.get_layer("conv2a").set_weights([w_a, b_a])
#         model.get_layer("conv2b").set_weights([w_b, b_b])

#         conv3w = weights_dic["conv3"][0]
#         conv3b = weights_dic["conv3"][1]
#         model.get_layer("conv3").set_weights([conv3w, conv3b])

#         conv4w = weights_dic["conv4"][0]
#         conv4b = weights_dic["conv4"][1]
#         w_a, w_b = np.split(conv4w, 2, axis=-1)
#         b_a, b_b = np.split(conv4b, 2, axis=-1)
#         model.get_layer("conv4a").set_weights([w_a, b_a])
#         model.get_layer("conv4b").set_weights([w_b, b_b])

#         conv5w = weights_dic["conv5"][0]
#         conv5b = weights_dic["conv5"][1]
#         w_a, w_b = np.split(conv5w, 2, axis=-1)
#         b_a, b_b = np.split(conv5b, 2, axis=-1)
#         model.get_layer("conv5a").set_weights([w_a, b_a])
#         model.get_layer("conv5b").set_weights([w_b, b_b])

#         fc6w = weights_dic["fc6"][0]
#         fc6b = weights_dic["fc6"][1]
#         model.get_layer("fc6").set_weights([fc6w, fc6b])

#         fc7w = weights_dic["fc7"][0]
#         fc7b = weights_dic["fc7"][1]
#         model.get_layer("fc7").set_weights([fc7w, fc7b])

#         # fc8w = weights_dic["fc8"][0]
#         # fc8b = weights_dic["fc8"][1]
#         # model.get_layer("fc8").set_weights([fc8w, fc8b])

#         print("Weights loaded.")

#     return model


def create_model(name, input_shape=(227, 227, 3), freeze_upto=15):
    base_model = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(227, 227, 3), pooling="avg"
    )
    for idx, layer in enumerate(base_model.layers):
        if idx < freeze_upto:
            layer.trainable = False
        else:
            layer.trainable = True

    x = layers.Dropout(0.3)(base_model.output)
    pred = layers.Dense(31, kernel_initializer=initializer)(x)
    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=pred, name=name)
    return model


ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    source_directory,
    labels="inferred",
    label_mode="int",  # categorical, binary
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
)


# def read_images(directory, batch_size, new_size):
#     ds = tf.keras.preprocessing.image_dataset_from_directory(
#         directory,
#         labels="inferred",
#         label_mode="int",  # categorical, binary
#         batch_size=batch_size,
#         image_size=(new_size, new_size),  # reshape if not in this size
#         shuffle=True,
#     )
#     return ds


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    image = image / 255.0

    return image, label


def augment_ds(image, label, prob=0.15):

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
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 2, 5)
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)

    # Rotating Images
    image = tf.cond(
        tf.random.uniform(shape=[], minval=0, maxval=1) < prob,
        lambda: tf.image.rot90(image, k=1),
        lambda: tf.image.rot90(image, k=3),
    )

    # Flipping Images
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label


# def augment(x, y):
#     image = tf.image.random_brightness(x, max_delta=0.05)
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)
#     return image, y


ds_train = (
    ds_train.map(preprocess).map(augment_ds).prefetch(tf.data.experimental.AUTOTUNE)
)

model = AlexNet()
# model = create_model("Testing")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=20, verbose=1)
