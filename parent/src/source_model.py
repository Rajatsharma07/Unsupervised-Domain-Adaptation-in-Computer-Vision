import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow import keras
import config as cn
from tensorflow.keras import models, layers


def source_resnet(input_shape):

    base_model = ResNet50(
        include_top=False, pooling=None, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = True
    predictions = layers.Conv2D(
        256, kernel_size=1, kernel_initializer=cn.initializer, name="Convolution1"
    )(base_model.output)
    # predictions = tf.keras.layers.GlobalAveragePooling2D()(predictions)
    # predictions = tf.keras.layers.ActivityRegularization(l1=0.001, l2=0.001)(predictions)

    model = models.Model(
        inputs=base_model.inputs, outputs=predictions, name="Source_Model"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
