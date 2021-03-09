import tensorflow as tf
from tensorflow.keras import models, layers
import src.config as cn
from src.loss import CORAL, coral_loss, kl_divergence
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import tempfile


def get_model(
    input_shape,
    prune,
    additional_loss,
    technique,
    num_classes=31,
    lambda_loss=0.75,
    prune_val=0.10,
):

    inputs = [
        tf.keras.layers.Input(shape=(input_shape)),
        tf.keras.layers.Input(shape=(input_shape)),
    ]

    if not technique:
        # Original Technique
        model = tf.keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=input_shape,
        )
        if prune:
            # Prune Target Model
            pruning_params = {
                "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                    prune_val, 0, end_step=-1, frequency=1
                )
            }
            model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

        source_op = model(inputs[0])
        target_op = model(inputs[1])

    else:
        # Our Technique
        source_model = tf.keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=input_shape,
        )
        target_model = tf.keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=input_shape,
        )

        if prune:
            # Prune Target Model
            pruning_params = {
                "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                    prune_val, 0, end_step=-1, frequency=1
                )
            }
            target_model = tfmot.sparsity.keras.prune_low_magnitude(
                target_model, **pruning_params
            )

        # Renaming Layers
        for layer in source_model.layers:
            layer._name = layer.name + str("_1")

        for layer in target_model.layers:
            layer._name = layer.name + str("_2")

        source_op = source_model(inputs[0])
        target_op = target_model(inputs[1])

    # Top Layer
    classifier = layers.Dropout(0.3)(source_op)
    classifier = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=cn.initializer,
        name="prediction",
    )(classifier)
    model = models.Model(inputs, classifier)

    # CORAL LOSS addition to the network
    additional_loss = cn.LOSS[additional_loss]

    additive_loss = additional_loss(
        source_output=source_op,
        target_output=target_op,
        percent_lambda=lambda_loss,
    )

    model.add_loss(additive_loss)
    model.add_metric(additive_loss, name="CORAL_loss")

    return model


def AlexNet(
    img_shape=(227, 227, 3), num_classes=31, weights="/content/bvlc_alexnet.npy"
):
    inputs = tf.keras.Input(img_shape)
    conv1 = conv2d_bn(
        x=inputs,
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
    drop6 = layers.Dropout(0.3)(fc6)
    fc7 = layers.Dense(4096, activation="relu", name="fc7")(drop6)
    drop7 = layers.Dropout(0.3)(fc7)
    # fc8 = layers.Dense(
    #     num_classes,
    #     name="fc8",
    #     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
    # )(drop7)
    model = models.Model(inputs, drop7)
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
