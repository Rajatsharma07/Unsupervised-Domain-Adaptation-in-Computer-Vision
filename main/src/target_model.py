from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2, l1_l2
import src.config as cn


def target_model(input_shape=(64, 64, 3)):
    """
    This method creates Destination model architecture.
    """
    in_2 = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        32,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv1_1",
        kernel_regularizer=l2(0.0001),
        bias_regularizer=l2(0.0001),
    )(in_2)
    x = layers.Conv2D(
        32,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv1_2",
        kernel_regularizer=l2(0.0001),
        bias_regularizer=l2(0.0001),
    )(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(
        64,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv2_1",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.Conv2D(
        64,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv2_2",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(
        128,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv3_1",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.Conv2D(
        128,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv3_2",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(
        256,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv4_1",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.Conv2D(
        256,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv4_2",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(
        256,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv5_1",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.Conv2D(
        256,
        kernel_size=3,
        kernel_initializer=cn.initializer,
        padding="same",
        name="conv5_2",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    out2 = layers.Dropout(0.4)(x)
    model = models.Model(inputs=in_2, outputs=out2, name="Target_Model")

    return model
