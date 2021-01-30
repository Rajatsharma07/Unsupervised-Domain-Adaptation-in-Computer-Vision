import tensorflow as tf
from tensorflow.keras import models, layers
import src.config as cn
from src.loss import CORAL, coral_loss
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import tempfile


def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ["kernel_regularizer"]:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), "tmp_weights.h5")
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def merged_model(
    input_shape,
    prune,
    num_classes=31,
    lambda_loss=0.75,
    additional_loss=CORAL,
):
    source_model = create_model("source_fe", input_shape)

    ip1 = tf.keras.Input(shape=(input_shape))
    ip2 = tf.keras.Input(shape=(input_shape))

    for layer in source_model.layers:
        layer._name = layer.name + str("_1")

    if prune:
        target_model = tfmot.sparsity.keras.prune_low_magnitude(
            create_model("target_fe", input_shape), **cn.pruning_params
        )
    else:
        target_model = create_model("target_fe", input_shape)

    for layer in target_model.layers:
        layer._name = layer.name + str("_2")

    op1 = source_model(ip1, training=False)
    op2 = target_model(ip2, training=False)

    # for idx, layer in enumerate(source_model.layers):
    #     if idx < freeze_upto:
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True

    # for idx, layer in enumerate(target_model.layers):
    #     if idx < freeze_upto:
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True

    x = layers.Dropout(0.4)(op1)
    prediction = tf.keras.layers.Dense(
        31, kernel_initializer=cn.initializer, name="prediction",
    )(x)
    model = models.Model([ip1, ip2], prediction)
    additive_loss = additional_loss(
        source_output=op1, target_output=op2, percent_lambda=lambda_loss,
    )

    model.add_loss(additive_loss)
    model.add_metric(additive_loss, name="domain_loss")
    return model


# def merged_model(
#     input_shape,
#     prune,
#     weights=cn.MODEL_PATH / "pretrained_weights/bvlc_alexnet.npy",
#     num_classes=31,
#     lambda_loss=0.75,
#     additional_loss=CORAL,
# ):
#     source_model = AlexNet(
#         img_shape=input_shape, num_classes=num_classes, weights=weights
#     )
#     source_model._name = "Source"
#     for layer in source_model.layers:
#         layer._name = layer.name + str("_1")

#     for layer in source_model.layers[:12]:
#         layer.trainable = False

#     if prune:
#         target_model = tfmot.sparsity.keras.prune_low_magnitude(
#             AlexNet(img_shape=input_shape, num_classes=num_classes, weights=weights),
#             **cn.pruning_params
#         )
#     else:
#         target_model = AlexNet(
#             img_shape=input_shape, num_classes=num_classes, weights=weights
#         )
#     target_model._name = "Target"

#     for layer in target_model.layers:
#         layer._name = layer.name + str("_2")

#     for layer in target_model.layers[:12]:
#         layer.trainable = False

#     prediction = layers.Dense(
#         num_classes,
#         kernel_initializer=tf.initializers.RandomNormal(0, 0.005),
#         name="prediction",
#     )(source_model.output)
#     model = models.Model(
#         [source_model.input, target_model.input], prediction, name="Full_Model"
#     )

#     additive_loss = additional_loss(
#         source_output=source_model.output,
#         target_output=target_model.output,
#         percent_lambda=lambda_loss,
#     )

#     model.add_loss(additive_loss)
#     model.add_metric(additive_loss, name="domain_loss")
#     return model


# def custom_alexnet(input_shape=(32, 32, 3)):
#     """
#     This method creates feature extractor model architecture.
#     """
#     model = models.Sequential(
#         [
#             layers.Conv2D(
#                 filters=96,
#                 kernel_size=(11, 11),
#                 strides=(4, 4),
#                 kernel_initializer=cn.initializer,
#                 input_shape=input_shape,
#             ),
#             layers.BatchNormalization(),
#             layers.Activation("relu"),
#             layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
#             layers.Conv2D(
#                 filters=256,
#                 kernel_size=(5, 5),
#                 strides=(1, 1),
#                 kernel_initializer=cn.initializer,
#                 padding="valid",
#             ),
#             layers.BatchNormalization(),
#             layers.Activation("relu"),
#             layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
#             layers.Conv2D(
#                 filters=384,
#                 kernel_size=(3, 3),
#                 strides=(1, 1),
#                 kernel_initializer=cn.initializer,
#                 padding="valid",
#             ),
#             layers.BatchNormalization(),
#             layers.Activation("relu"),
#             layers.Conv2D(
#                 filters=384,
#                 kernel_size=(1, 1),
#                 strides=(1, 1),
#                 kernel_initializer=cn.initializer,
#                 padding="valid",
#             ),
#             layers.BatchNormalization(),
#             layers.Activation("relu"),
#             layers.Conv2D(
#                 filters=256,
#                 kernel_size=(1, 1),
#                 strides=(1, 1),
#                 kernel_initializer=cn.initializer,
#                 padding="valid",
#             ),
#             layers.BatchNormalization(),
#             layers.Activation("relu"),
#             layers.MaxPooling2D(
#                 pool_size=(3, 3),
#                 strides=(2, 2),
#                 padding="valid",
#             ),
#             layers.Dropout(0.3),
#             layers.Flatten(),
#         ]
#     )

#     return model


# def custom_Lenet(input_shape=(32, 32, 3)):
#     """
#     This method creates feature extractor model architecture.
#     """
#     model = models.Sequential(
#         [
#             layers.Conv2D(
#                 filters=64,
#                 kernel_size=5,
#                 kernel_initializer=cn.initializer,
#                 input_shape=input_shape,
#             ),
#             layers.BatchNormalization(),
#             layers.Activation("relu"),
#             layers.MaxPooling2D(pool_size=2, strides=2),
#             layers.Conv2D(
#                 filters=128,
#                 kernel_size=5,
#                 kernel_initializer=cn.initializer,
#             ),
#             layers.BatchNormalization(),
#             layers.Activation("relu"),
#             layers.MaxPooling2D(pool_size=2, strides=2),
#             layers.Flatten(),
#             layers.Dense(1024, kernel_initializer=cn.initializer, name="fc3"),
#             layers.BatchNormalization(),
#             layers.Activation("relu"),
#             layers.Dropout(0.3),
#             layers.Dense(64, kernel_initializer=cn.initializer, name="fc4"),
#             layers.BatchNormalization(),
#             layers.Activation("relu"),
#             layers.Dropout(0.3),
#         ]
#     )

#     return model


def AlexNet(
    img_shape=(227, 227, 3), num_classes=31, weights="/content/bvlc_alexnet.npy"
):
    """[summary]

    Args:
        img_shape (tuple, optional): [description]. Defaults to (227, 227, 3).
        num_classes (int, optional): [description]. Defaults to 1000.
        weights (str, optional): [description]. Defaults to "/content/bvlc_alexnet.npy".

    Returns:
        [type]: [description]
    """
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


def create_model(name, input_shape=(299, 299, 3)):
    base_model = tf.keras.applications.Xception(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    model = tf.keras.models.Model(
        inputs=base_model.inputs, outputs=base_model.outputs, name=name
    )
    return model
