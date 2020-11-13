import os, sys

print(os.getcwd())
base_dir = os.getcwd()
# os.chdir(base_dir)
train_path = "/content/drive/My Drive/Thesis/datasets/keras_mnistm.pkl"
os.getcwd()

sys.path.append("/content/drive/My Drive/Thesis/")

from tensorflow.keras import layers, backend as K
import my_utils as methods
import datetime
import pickle
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical, plot_model
from keras.callbacks import CSVLogger

tf.random.set_seed(100)
tf.keras.backend.clear_session()  # For easy reset of notebook state.
initializer = tf.keras.initializers.he_normal()
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2, l1, l1_l2

tf.keras.backend.clear_session()
mnist = tf.keras.datasets.mnist
(mnistx_train, mnisty_train), (mnistx_test, mnisty_test) = mnist.load_data()

print(
    f"mnistx_train shape: {mnistx_train.shape}, mnisty_train shape : {mnisty_train.shape}"
)
print(
    f"mnistx_test shape: {mnisty_test.shape}, mnisty_test shape : {mnisty_test.shape}"
)

# Preprocessing
mnistx_train = tf.image.resize(
    tf.reshape(mnistx_train, shape=[-1, 28, 28, 1]),
    [32, 32],
    method="nearest",
    preserve_aspect_ratio=False,
    antialias=True,
    name=None,
)
mnistx_test = tf.image.resize(
    tf.reshape(mnistx_test, shape=[-1, 28, 28, 1]),
    [32, 32],
    method="nearest",
    preserve_aspect_ratio=False,
    antialias=True,
    name=None,
)

mnistx_train = tf.image.grayscale_to_rgb(mnistx_train)
mnistx_test = tf.image.grayscale_to_rgb(mnistx_test)

methods.display_dataset(mnistx_test, False)


mnistm_path = train_path

with open(mnistm_path, "rb") as f:
    mnistm_dataset = pickle.load(f, encoding="bytes")

print(mnistm_dataset.keys())
mnistmx_train = mnistm_dataset[b"train"]
mnistmx_test = mnistm_dataset[b"test"]
mnistmx_train.shape, mnistmx_test.shape

## Preprocessing for Keras inbuilt models - ResNet 50
# Resizing (minimum 32*32)
mnistmx_train = tf.image.resize(
    mnistmx_train,
    [32, 32],
    method="nearest",
    preserve_aspect_ratio=False,
    antialias=True,
    name=None,
)
mnistmx_test = tf.image.resize(
    mnistmx_test,
    [32, 32],
    method="nearest",
    preserve_aspect_ratio=False,
    antialias=True,
    name=None,
)

# # Visualize MNISTM
methods.display_dataset(mnistmx_test, False)

"""# Source Model

## Summary
"""


def source_ResNet(img_shape, num_classes):

    inputs = layers.Input(shape=img_shape)
    base_model = ResNet50(
        include_top=False, pooling=None, weights="imagenet", input_shape=img_shape
    )
    base_model.trainable = True
    predictions = layers.Conv2D(
        256, kernel_size=1, kernel_initializer=initializer, name="Convolution1"
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


# Create source model
tf.keras.backend.clear_session()
source_mdl = None
source_mdl = source_ResNet((32, 32, 3), 10)

source_path = os.path.join(base_dir, "experiments")

source_mdl.summary()

# plot_model(source_mdl, source_path + "/source_ResNet.png", show_shapes=True)

# model.save_weights(base_dir+"/usps_target/model/usps_model.hdf5")
# source_model.save(os.path.join(source_path,"model/mnistm_model.h5"))

"""# Target Model"""


def target_model(input_shape=(64, 64, 3)):
    """
    This method creates Destination model architecture.
    """
    in_2 = Input(shape=input_shape)

    x = layers.Conv2D(
        32,
        kernel_size=3,
        kernel_initializer=initializer,
        padding="same",
        name="conv1_1",
        kernel_regularizer=l2(0.0001),
        bias_regularizer=l2(0.0001),
    )(in_2)
    x = layers.Conv2D(
        32,
        kernel_size=3,
        kernel_initializer=initializer,
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
        kernel_initializer=initializer,
        padding="same",
        name="conv2_1",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.Conv2D(
        64,
        kernel_size=3,
        kernel_initializer=initializer,
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
        kernel_initializer=initializer,
        padding="same",
        name="conv3_1",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.Conv2D(
        128,
        kernel_size=3,
        kernel_initializer=initializer,
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
        kernel_initializer=initializer,
        padding="same",
        name="conv4_1",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.Conv2D(
        256,
        kernel_size=3,
        kernel_initializer=initializer,
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
        kernel_initializer=initializer,
        padding="same",
        name="conv5_1",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.Conv2D(
        256,
        kernel_size=3,
        kernel_initializer=initializer,
        padding="same",
        name="conv5_2",
        kernel_regularizer=l1_l2(0.0001),
        bias_regularizer=l1_l2(0.0001),
    )(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    out2 = layers.Dropout(0.4)(x)

    # out2 = GlobalAveragePooling2D()(out2)

    model = models.Model(inputs=in_2, outputs=out2, name="Custom_VGG_Model")
    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=["accuracy"],
    # )
    return model


target_mdl = target_model(input_shape=(32, 32, 3))
target_mdl.summary()

target_path = source_path
# plot_model(target_mdl, target_path + "/target_VGG.png", show_shapes=True)

"""# Merged Network"""


# class CustomFit(keras.Model):
#     def __init__(self, model):
#         super(CustomFit, self).__init__()
#         self.model = model

#     def compile(self, optimizer, loss):
#         super(CustomFit, self).compile()
#         self.optimizer = optimizer
#         self.loss = loss

#     def train_step(self, data):
#         x, y = data

#         with tf.GradientTape() as tape:
#             # Caclulate predictions
#             y_pred = self.model(x, training=True)

#             # Loss
#             loss = self.loss(y, y_pred)

#         # Gradients
#         training_vars = self.trainable_variables
#         gradients = tape.gradient(loss, training_vars)

#         # Step with optimizer
#         self.optimizer.apply_gradients(zip(gradients, training_vars))
#         acc_metric.update_state(y, y_pred)

#         return {"loss": loss, "accuracy": acc_metric.result()}

#     def test_step(self, data):
#         # Unpack the data
#         x, y = data

#         # custom_model = Sequential()
#         # custom_model.add(self.model.layers[3])
#         # custom_model.add(self.model.layers[6])
#         # # custom_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
#         # y_pred = custom_model(x, training=False)

#         # idx = 3  # index of desired layer
#         # input_shape = self.model.layers[idx].get_input_shape_at(0) # get the input shape of desired layer
#         # layer_input = Input(shape=input_shape) # a new input tensor to be able to feed the desired layer

#         # # create the new nodes for each layer in the path
#         # op = self.model.layers[6](layer_input)
#         # # create the model
#         # op_mdl = Model(layer_input, op)
#         # op_mdl.compile()
#         # y_pred = op_mdl.predict(y_inter)

#         # Compute predictions
#         y_pred = self.model(x, training=False)

#         # Updates the metrics tracking the loss
#         loss = self.loss(y, y_pred)

#         # Update the metrics.
#         acc_metric.update_state(y, y_pred)
#         return {"loss": loss, "accuracy": acc_metric.result()}


acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

tf.keras.backend.clear_session()  # For easy reset of notebook state.


def merged_network(
    input_shape, source_model, target_model, num_classes=10, percent=0.25
):
    """
    This method creates merged network having concatenation of Source & Destination Model.

    """
    in_1 = layers.Input(shape=input_shape)
    source_model = source_model(in_1)
    source_model = models.Model(in_1, source_model, name="Source_Model")
    source_model.trainable = True
    source_model.summary()

    in_2 = layers.Input(shape=input_shape)
    target_model = target_model(in_2, training=True)
    target_model = models.Model(in_2, target_model, name="Target_Model")
    target_model.trainable = True
    target_model.summary()

    # ## Concatenation of Networks
    # in_3 = Input((source_model.output.shape[1]+target_model.output.shape[1]))
    # # concatenated = concatenate([source_model, target_model])
    # x = BatchNormalization(name="bn_top")(in_3)
    # x = Activation("relu")(x)
    # x = Dropout(0.4)(x)
    # out_concat = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(x)
    # concat_model = Model(in_3, out_concat, name="Concatenated_Model")
    # concat_model.summary()

    # concat = tf.keras.layers.Concatenate()([source_model.output, target_model.output])
    # combi = concat_model(concat)

    # merged_model = Model([source_model.input, target_model.input], combi)

    # # kl_loss = -0.5 * tf.reduce_mean(target_model.output - tf.square(source_model.output) - tf.exp(target_model.output) + 1)
    # # merged_model.add_loss(kl_loss)

    # merged_model.compile(
    #     optimizer=keras.optimizers.Nadam(learning_rate=0.0001),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=["accuracy"],
    # )

    concat = layers.Concatenate()([source_model.output, target_model.output])
    ## Concatenation of Networks
    # in_3 = Input((source_model.output.shape[1]+target_model.output.shape[1]))
    # in_3 = Input((target_model.output.shape[1], target_model.output.shape[2], source_model.output.shape[3]+target_model.output.shape[3]))
    concat = layers.Conv2D(
        256, kernel_size=1, kernel_initializer=initializer, name="DownsampleConvolution"
    )(concat)

    classifier_input = layers.Input(
        (
            target_model.output.shape[1],
            target_model.output.shape[2],
            target_model.output.shape[3],
        )
    )
    x = layers.GlobalAveragePooling2D()(classifier_input)
    x = layers.BatchNormalization(name="bn_top1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, kernel_initializer=initializer)(x)
    x = layers.BatchNormalization(name="bn_top2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    prediction = layers.Dense(num_classes, kernel_initializer=initializer)(x)

    classifier_model = models.Model(classifier_input, prediction, name="Classifier")
    classifier_model.summary()

    final_output = classifier_model(concat)

    merged_model = models.Model([source_model.input, target_model.input], final_output)

    source_scores = tf.reshape(source_model.output, [-1, 256])
    target_scores = tf.reshape(target_model.output, [-1, 256])

    source_batch_size = tf.cast(tf.shape(source_scores)[0], tf.float32)
    target_batch_size = tf.cast(tf.shape(target_scores)[0], tf.float32)
    d = tf.cast(tf.shape(source_scores)[1], tf.float32)

    # Source covariance
    xm = source_scores - tf.reduce_mean(source_scores, 0, keepdims=True)
    xc = tf.matmul(tf.transpose(xm), xm) / source_batch_size

    # Target covariance
    xmt = target_scores - tf.reduce_mean(target_scores, 0, keepdims=True)
    xct = tf.matmul(tf.transpose(xmt), xmt) / target_batch_size

    coral_loss_1 = tf.reduce_sum(tf.multiply((xc - xct), (xc - xct)))
    coral_loss = coral_loss_1 / (4 * d * d)
    merged_model.add_loss(percent * coral_loss)

    return merged_model


tf.keras.backend.clear_session()
"""
In Depth Merged Network summary
"""
m1 = None
m1 = merged_network(
    (32, 32, 3), source_model=source_mdl, target_model=target_mdl, percent=1
)

methods.model_layers(m1)

# plot_model(m1, show_shapes=True)


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.validation_data = val_data

    def on_train_begin(self, logs={}):
        self.scce = []

    def on_epoch_end(self, epoch, logs={}):
        # val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        # val_targ = self.validation_data[1]
        # _val_scce = scce(val_targ, val_predict).numpy()
        # self.scce.append(_val_scce)

        # create the new nodes for each layer in the path
        x = self.model.layers[3](self.validation_data[0])
        y_pred = self.model.layers[6](x)

        # ip1 = keras.Input(shape=(32,32,3))
        # # ip1 = self.model.layers[3].input_shape[1:]
        # op1 = self.model.layers[3](ip1)
        # # create the model
        # op_mdl1 = Model(ip1, op1)
        # o1 = op_mdl1(self.validation_data[0])
        # print(o1[0])

        # create the new nodes for each layer in the path
        # ip2 = keras.Input(shape=(1,1,256))
        # op2 = self.model.layers[6](ip2)
        # op_mdl2 = Model(ip2, op2)

        # loss, acc = mdl.evaluate(self.validation_data[0], self.validation_data[1], verbose=0)
        loss = loss_fn(self.validation_data[1], y_pred)
        # print('\n')
        # # print('SCCE score -->',float(_val_scce))
        # # loss, acc = self.model.evaluate(self.validation_data[0], self.validation_data[1], verbose=0)
        acc_metric.update_state(self.validation_data[1], y_pred)
        print(f"My_loss: {loss}, My_accuracy: {acc_metric.result()}")
        return {"My_loss": loss, "My_accuracy": acc_metric.result()}


m1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy(name="my_accuracy")


methods.model_layers(m1)

## FINAL MERGED NETWORK
m1_path = os.path.join(base_dir, source_path)
# plot_model(m1, m1_path + "/m1.png", show_shapes=True)


"""# Grey MNIST to MNISTM, Training"""


def callbacks_fn(model_path):
    checkpoint_path = os.path.join(
        model_path, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    )
    print(checkpoint_path)

    csv_logger = CSVLogger("log.csv", append=True, separator=";")

    logdir = os.path.join(base_dir, "logs")
    logdir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(logdir)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
        monitor="val_accuracy",
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=1, min_lr=0.000001
    )
    return csv_logger, cp_callback, tensorboard_callback, reduce_lr_callback, logdir


(
    csv_logger,
    cp_callback,
    tensorboard_callback,
    reduce_lr_callback,
    logdir,
) = callbacks_fn(m1_path)

"""## 50% CORAL"""

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir "/content/drive/My Drive/Thesis/logs/20201101-152555"

# Shuffling Target Training Set Classes
np.random.seed(121)
index_shuffled = np.arange(60000)
np.random.shuffle(index_shuffled)

mnistx_train = np.array(mnistx_train)
# mnistx_test=np.array(mnistx_test)

mnistx_train = mnistx_train[index_shuffled]
mnisty_train = (np.array(mnisty_train))[index_shuffled]
# shuffle = np.arange(10000)
# np.random.shuffle(shuffle)
# mnistx_test = mnistx_test[shuffle]

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title("{}".format(mnisty_train[i + 8600]))
    plt.imshow(mnistx_train[i + 8600].astype(np.uint8))
plt.show()

mnistx_train_normalized = tf.cast(mnistx_train, tf.float32)
mnistx_train_normalized = tf.keras.applications.resnet50.preprocess_input(
    mnistx_train_normalized
)

mnistx_test_normalized = tf.cast(mnistx_test, tf.float32)
mnistx_test_normalized = tf.keras.applications.resnet50.preprocess_input(
    mnistx_test_normalized
)


## Casting and Scaling dataset
mnistmx_train_normalized = tf.cast(mnistmx_train, tf.float32)
mnistmx_train_normalized = tf.keras.applications.resnet50.preprocess_input(
    mnistmx_train_normalized
)

mnistmx_test_normalized = tf.cast(mnistmx_test, tf.float32)
mnistmx_test_normalized = tf.keras.applications.resnet50.preprocess_input(
    mnistmx_test_normalized
)

metrics = Metrics((mnistmx_test_normalized, mnisty_test))

print(mnistx_train_normalized.shape)
print(mnisty_train.shape)

# tf.keras.backend.clear_session()
m1_hist = None
m1_hist = m1.fit(
    x=[mnistx_train_normalized, mnistmx_train_normalized],
    y=mnisty_train,
    validation_data=([mnistmx_test_normalized, mnistmx_test_normalized], mnisty_test),
    epochs=5,
    verbose=1,
    callbacks=[
        cp_callback,
        tensorboard_callback,
        reduce_lr_callback,
        csv_logger,
        metrics,
    ],
    batch_size=128,
)


methods.loss_accuracy_plots(
    m1_hist.history["accuracy"],
    m1_hist.history["val_accuracy"],
    m1_hist.history["loss"],
    m1_hist.history["val_loss"],
)

# m1.save(os.path.join(m1_path, "m1_model.h5"))

"""## Evaluation"""

methods.test_accuracy(
    m1, [mnistmx_test_normalized, mnistmx_test_normalized], mnisty_test
)


methods.test_accuracy(
    m1, [mnistmx_test_normalized, mnistmx_test_normalized], mnisty_test
)
