import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.regularizers import l1_l2
import src.config as cn
from src.loss import CORAL


class DeepCORAL(keras.Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(DeepCORAL, self).__init__()
        self.feature_extractor = upper_model(input_shape=input_shape)
        self.classifier = classifier_model(
            shape=self.feature_extractor.output[0].shape[1], num_classes=num_classes
        )

    def call(self, inputs, training=False):
        source, target = self.feature_extractor(inputs, training=training)
        source_output = self.classifier(source, training=training)

        loss = CORAL(source, target)
        self.add_loss(loss)

        if training:
            self.add_metric(loss, name="CORAL_Loss", aggregation="mean")

        return source_output


def classifier_model(shape, num_classes=10):
    inputs = keras.Input(shape)
    x = layers.BatchNormalization(name="bn_top1")(inputs)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(
        512,
        kernel_initializer=cn.initializer,
    )(x)
    x = layers.BatchNormalization(name="bn_top2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    prediction = layers.Dense(
        num_classes,
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005),
    )(x)
    classifier_model = keras.Model(inputs, prediction, name="Classifier")
    classifier_model.summary()
    return classifier_model


def upper_model(input_shape):

    model_input = [keras.Input(shape=input_shape), keras.Input(shape=input_shape)]

    source_model = Custom_VGG16([32, 64, 128, 256, 256])
    source_model._name = "Source"
    for layer in source_model.layers:
        layer._name = layer.name + str("_1")

    target_model = Custom_VGG16([32, 64, 128, 256, 256])
    target_model._name = "Target"
    for layer in target_model.layers:
        layer._name = layer.name + str("_2")

    source_model.trainable = True
    target_model.trainable = True

    source = source_model(model_input[0])
    target = target_model(model_input[1])

    model = models.Model(
        inputs=model_input, outputs=[source, target], name="Dual_Network"
    )
    return model


class CNN_Block(layers.Layer):
    def __init__(self, out_channels, name, kernel_size=3):
        super(CNN_Block, self).__init__()
        self.conv1 = self.conv3 = layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            name=name + "_1",
            kernel_initializer=cn.initializer,
            padding="same",
            # kernel_regularizer=l1_l2(0.0001),
            # bias_regularizer=l1_l2(0.0001),
        )
        self.conv2 = layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            name=name + "_2",
            kernel_initializer=cn.initializer,
            padding="same",
            # kernel_regularizer=l1_l2(0.0001),
            # bias_regularizer=l1_l2(0.0001),
        )
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor, training=training)
        x = self.conv2(x, training=training)
        x = self.bn(x, training=training)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.3)(x)
        return x


class Custom_VGG16(keras.Model):
    def __init__(self, channels):
        super(Custom_VGG16, self).__init__()
        self.channels = channels
        self.cnn1 = CNN_Block(channels[0], "conv1")
        self.cnn2 = CNN_Block(channels[1], "conv2")
        self.cnn3 = CNN_Block(channels[2], "conv3")
        self.cnn4 = CNN_Block(channels[3], "conv4")
        self.cnn5 = CNN_Block(channels[4], "conv5")

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(x, training=training)
        x = self.cnn4(x, training=training)
        x = self.cnn5(x, training=training)
        x = layers.GlobalAveragePooling2D()(x)
        return x

    def model(self):
        inputs = keras.Input(shape=(32, 32, 3))
        return keras.Model(inputs=inputs, outputs=self.call(inputs))
