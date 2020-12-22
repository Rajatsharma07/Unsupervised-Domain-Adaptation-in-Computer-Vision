import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import models, layers
import src.config as cn


def CORAL(source_output, target_output, percent_lambda=0.5):

    source_batch_size = tf.cast(tf.shape(source_output)[0], tf.float32)
    target_batch_size = tf.cast(tf.shape(target_output)[0], tf.float32)
    d = tf.cast(tf.shape(source_output)[1], tf.float32)

    # Source covariance
    xm = source_output - tf.reduce_mean(source_output, 0, keepdims=True)
    xc = tf.matmul(tf.transpose(xm), xm) / source_batch_size

    # Target covariance
    xmt = target_output - tf.reduce_mean(target_output, 0, keepdims=True)
    xct = tf.matmul(tf.transpose(xmt), xmt) / target_batch_size

    # Frobenius norm
    loss = tf.sqrt(tf.reduce_sum(tf.multiply((xc - xct), (xc - xct))))
    loss = loss / (4 * d * d)
    loss = percent_lambda * loss
    # model.add_loss(loss)
    return loss


class DeepCORAL(keras.Model):
    def __init__(self, input_shape=(32, 32, 3), is_pretrained=False, num_classes=10):
        super(DeepCORAL, self).__init__()
        self.feature_extractor = upper_model(
            input_shape=input_shape, is_pretrained=is_pretrained
        )
        self.classifier = classifier_model(
            shape=self.feature_extractor.output[0].shape[1], num_classes=num_classes
        )

    def call(self, inputs, training=None):
        source, target = self.feature_extractor(inputs)
        source_output = self.classifier(source)

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
        1024,
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


def upper_model(input_shape, is_pretrained=False):

    model_input = [keras.Input(shape=input_shape), keras.Input(shape=input_shape)]

    if is_pretrained:
        base_model = VGG16(
            include_top=False,
            pooling="avg",
            weights="imagenet",
            input_shape=input_shape,
        )
    else:
        base_model = VGG16(include_top=False, pooling="avg", input_shape=input_shape)

    if is_pretrained:
        base_model1 = VGG16(
            include_top=False,
            pooling="avg",
            weights="imagenet",
            input_shape=input_shape,
        )
    else:
        base_model1 = VGG16(include_top=False, pooling="avg", input_shape=input_shape)

    base_model._name = "SourceVGG"
    base_model1._name = "TargetVGG"

    # for layer in base_model.layers:
    #     print(layer._name)

    for layer in base_model1.layers:
        layer._name = layer.name + str("_2")
        print(layer._name)

    base_model.trainable = True
    base_model1.trainable = True

    source = base_model(model_input[0])
    target = base_model1(model_input[1])

    model = models.Model(
        inputs=model_input, outputs=[source, target], name="Dual_Network"
    )
    return model
