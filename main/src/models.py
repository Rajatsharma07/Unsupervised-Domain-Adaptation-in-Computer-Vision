import tensorflow as tf
from tensorflow.keras import models, layers
import src.config as cn
from src.loss import CORAL
import tensorflow_model_optimization as tfmot


# def merged_model(
#     input_shape, prune, num_classes=10, lambda_loss=0.25, additional_loss=CORAL
# ):
#     source_model = custom_vgg16(input_shape)
#     source_model._name = "Source"
#     for layer in source_model.layers:
#         layer._name = layer.name + str("_source")

#     if prune:
#         target_model = tfmot.sparsity.keras.prune_low_magnitude(
#             custom_vgg16(input_shape), **cn.pruning_params
#         )
#     else:
#         target_model = custom_vgg16(input_shape)

#     target_model._name = "Target"
#     for layer in target_model.layers:
#         layer._name = layer.name + str("_target")

#     source_model.trainable = True
#     target_model.trainable = True

#     x = layers.BatchNormalization(name="bn_top1")(source_model.output)
#     x = layers.Activation("relu")(x)
#     x = layers.Dense(4096, kernel_initializer=cn.initializer)(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Dense(4096, kernel_initializer=cn.initializer)(x)
#     x = layers.Activation("relu")(x)
#     prediction = layers.Dense(
#         num_classes,
#         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
#     )(x)
#     model = models.Model(
#         [source_model.input, target_model.input], prediction, name="Full_Model"
#     )

#     additive_loss = additional_loss(
#         source_output=source_model.output,
#         target_output=target_model.output,
#         percent_lambda=lambda_loss,
#     )

#     model.add_loss(additive_loss)
#     model.add_metric(additive_loss, name="Loss2", aggregation="mean")
#     return model


# def merged_model1(
#     input_shape, prune, num_classes=10, lambda_loss=0.25, additional_loss=CORAL
# ):
#     source_model = custom_alexnet(input_shape)
#     source_model._name = "Source"
#     for layer in source_model.layers:
#         layer._name = layer.name + str("_source")

#     if prune:
#         target_model = tfmot.sparsity.keras.prune_low_magnitude(
#             custom_alexnet(input_shape), **cn.pruning_params
#         )
#     else:
#         target_model = custom_alexnet(input_shape)

#     target_model._name = "Target"
#     for layer in target_model.layers:
#         layer._name = layer.name + str("_target")

#     source_model.trainable = True
#     target_model.trainable = True

#     x = layers.Dense(4096, kernel_initializer=cn.initializer)(source_model.output)
#     x = layers.BatchNormalization(name="bn_top2")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.Dense(4096, kernel_initializer=cn.initializer)(x)
#     x = layers.BatchNormalization(name="bn_top3")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Dropout(0.3)(x)
#     prediction = layers.Dense(
#         num_classes,
#         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
#     )(x)
#     model = models.Model(
#         [source_model.input, target_model.input], prediction, name="Full_Model"
#     )

#     additive_loss = additional_loss(
#         source_output=source_model.output,
#         target_output=target_model.output,
#         percent_lambda=lambda_loss,
#     )

#     model.add_loss(additive_loss)
#     model.add_metric(additive_loss, name="DomainLoss", aggregation="mean")
#     return model


def merged_model(
    input_shape, prune, num_classes=10, lambda_loss=0.25, additional_loss=CORAL
):
    source_model = custom_Lenet(input_shape)
    source_model._name = "Source"
    for layer in source_model.layers:
        layer._name = layer.name + str("_1")

    if prune:
        target_model = tfmot.sparsity.keras.prune_low_magnitude(
            custom_Lenet(input_shape), **cn.pruning_params
        )
    else:
        target_model = custom_Lenet(input_shape)

    target_model._name = "Target"
    for layer in target_model.layers:
        layer._name = layer.name + str("_2")

    source_model.trainable = True
    target_model.trainable = True

    # x = layers.Dense(1024, kernel_initializer=cn.initializer, name="fc3")(
    #     source_model.output
    # )
    # x = layers.BatchNormalization(name="bn_top1")(x)
    # x = layers.Activation("relu")(x)
    # x = layers.Dropout(0.3)(x)
    # x = layers.Dense(64, kernel_initializer=cn.initializer, name="fc4")(x)
    # x = layers.BatchNormalization(name="bn_top2")(x)
    # x = layers.Activation("relu")(x)
    # x = layers.Dropout(0.3)(x)
    prediction = layers.Dense(
        num_classes, kernel_initializer=cn.initializer, name="prediction"
    )(source_model.output)
    model = models.Model(
        [source_model.input, target_model.input], prediction, name="Full_Model"
    )

    additive_loss = additional_loss(
        source_output=source_model.output,
        target_output=target_model.output,
        percent_lambda=lambda_loss,
    )

    model.add_loss(additive_loss)
    model.add_metric(additive_loss, name="domain_loss", aggregation="mean")
    return model


# def custom_vgg16(input_shape=(32, 32, 3)):
#     """
#     This method creates feature extractor model architecture.
#     """
#     inputs = keras.Input(shape=input_shape)

#     x = layers.Conv2D(
#         32,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv1_1",
#     )(inputs)
#     x = layers.Conv2D(
#         32,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv1_2",
#     )(x)
#     x = layers.BatchNormalization(name="bn1")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Dropout(0.3)(x)

#     x = layers.Conv2D(
#         64,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv2_1",
#     )(x)
#     x = layers.Conv2D(
#         64,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv2_2",
#     )(x)
#     x = layers.BatchNormalization(name="bn2")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Dropout(0.3)(x)

#     x = layers.Conv2D(
#         128,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv3_1",
#     )(x)
#     x = layers.Conv2D(
#         128,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv3_2",
#     )(x)
#     x = layers.BatchNormalization(name="bn3")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Dropout(0.3)(x)

#     x = layers.Conv2D(
#         256,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv4_1",
#     )(x)
#     x = layers.Conv2D(
#         256,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv4_2",
#     )(x)
#     x = layers.BatchNormalization(name="bn4")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Dropout(0.3)(x)

#     x = layers.Conv2D(
#         256,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv5_1",
#     )(x)
#     x = layers.Conv2D(
#         256,
#         kernel_size=3,
#         kernel_initializer=cn.initializer,
#         padding="same",
#         name="conv5_2",
#     )(x)
#     x = layers.BatchNormalization(name="bn5")(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Dropout(0.3)(x)
#     outputs = layers.GlobalAveragePooling2D()(x)
#     model = models.Model(inputs=inputs, outputs=outputs)

#     return model


def custom_alexnet(input_shape=(32, 32, 3)):
    """
    This method creates feature extractor model architecture.
    """
    model = models.Sequential(
        [
            layers.Conv2D(
                filters=96,
                kernel_size=(11, 11),
                strides=(4, 4),
                kernel_initializer=cn.initializer,
                input_shape=input_shape,
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Conv2D(
                filters=256,
                kernel_size=(5, 5),
                strides=(1, 1),
                kernel_initializer=cn.initializer,
                padding="valid",
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                kernel_initializer=cn.initializer,
                padding="valid",
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(
                filters=384,
                kernel_size=(1, 1),
                strides=(1, 1),
                kernel_initializer=cn.initializer,
                padding="valid",
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(
                filters=256,
                kernel_size=(1, 1),
                strides=(1, 1),
                kernel_initializer=cn.initializer,
                padding="valid",
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
            ),
            layers.Dropout(0.3),
            layers.Flatten(),
        ]
    )

    return model


def custom_Lenet(input_shape=(32, 32, 3)):
    """
    This method creates feature extractor model architecture.
    """
    model = models.Sequential(
        [
            layers.Conv2D(
                filters=64,
                kernel_size=5,
                kernel_initializer=cn.initializer,
                input_shape=input_shape,
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(
                filters=128,
                kernel_size=5,
                kernel_initializer=cn.initializer,
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Flatten(),
            layers.Dense(1024, kernel_initializer=cn.initializer, name="fc3"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.3),
            layers.Dense(64, kernel_initializer=cn.initializer, name="fc4"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.3),
        ]
    )

    return model
