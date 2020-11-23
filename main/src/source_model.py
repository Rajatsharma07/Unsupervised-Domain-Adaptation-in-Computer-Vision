from tensorflow.keras.applications.resnet50 import ResNet50
import config as cn
from tensorflow.keras import models, layers


def source_resnet(input_shape):
    base_model = ResNet50(
        include_top=False, pooling=None, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = True
    predictions = layers.Conv2D(
        256,
        kernel_size=1,
        kernel_initializer=cn.initializer,
        name="Conv_Source",
        activation="relu",
    )(base_model.output)
    model = models.Model(
        inputs=base_model.inputs, outputs=predictions, name="Source_Model"
    )
    return model
