from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import src.config as cn
from src.loss import coral_loss
import datetime
import os
from tensorflow.keras.callbacks import CSVLogger
from pathlib import Path


def merged_network(
    input_shape,
    source_model,
    target_model,
    additional_loss=coral_loss,
    num_classes=10,
    percent=0.25,
    feature_extractor_shape=256,
):
    """
    This method creates merged network having concatenation of Source & Target Models.

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

    concat = layers.Concatenate()([source_model.output, target_model.output])
    concat = layers.Conv2D(
        feature_extractor_shape,
        kernel_size=1,
        kernel_initializer=cn.initializer,
        name="DownsampleConvolution",
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
    x = layers.Dense(64, kernel_initializer=cn.initializer)(x)
    x = layers.BatchNormalization(name="bn_top2")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    prediction = layers.Dense(
        num_classes, kernel_initializer=cn.initializer, activation="softmax"
    )(x)

    classifier_model = models.Model(classifier_input, prediction, name="Classifier")
    classifier_model.summary()

    final_output = classifier_model(concat)

    merged_model = models.Model([source_model.input, target_model.input], final_output)

    source_scores = tf.reshape(source_model.output, [-1, feature_extractor_shape])
    target_scores = tf.reshape(target_model.output, [-1, feature_extractor_shape])

    coral_loss = additional_loss(
        model=merged_model,
        source_output=source_scores,
        target_output=target_scores,
        percent_lambda=percent,
    )
    merged_model.add_metric(coral_loss, name="coral_loss", aggregation="mean")

    return merged_model


class Custom_Eval(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.validation_data = val_data
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(
            name="custom_accuracy_metric"
        )

    def on_epoch_begin(self, epoch, logs):
        self.custom_loss = []
        self.custom_acc = []

    def on_epoch_end(self, epoch, logs):
        for batch_idx, (x_batch, y_batch) in enumerate(self.validation_data):
            x = self.model.layers[3](x_batch, training=False)
            y_pred = self.model.layers[6](x, training=False)

            loss = self.loss_fn(y_batch, y_pred)
            self.custom_loss.append(loss.numpy())

            self.acc_metric.update_state(y_batch, y_pred)
            self.custom_acc.append(self.acc_metric.result().numpy())
            self.acc_metric.reset_states()

        total_val_loss = tf.reduce_mean(self.custom_loss).numpy()
        total_val_acc = tf.reduce_mean(self.custom_acc).numpy()
        # Writing to CSV logs
        logs["custom_accuracy"] = total_val_acc
        logs["custom_loss"] = total_val_loss
        # Writing to tensorboad logs
        tf.summary.scalar("Custom Evaluation loss", data=total_val_loss, step=epoch)
        tf.summary.scalar(
            "Custom Evaluation Accuracy",
            data=total_val_acc,
            step=epoch,
        )
        print(
            "custom_accuracy --> %.3f" % (float(total_val_acc),),
            f"custom_loss --> {total_val_loss}",
        )


def callbacks_fn(params):

    my_dir = (
        str(params["combination"])
        + "_"
        + params["source_model"]
        + "_"
        + str(params["sample_seed"])
    )
    print(f"Created directory: {my_dir}")
    callback_list = []

    """Tensorboard Callback """
    assert os.path.exists(cn.LOGS_DIR), "LOGS_DIR doesn't exist"
    tb_logdir = os.path.join(cn.LOGS_DIR, my_dir)
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)
    assert os.path.exists(tb_logdir), "tb_logdir doesn't exist"
    tb_logdir = os.path.join(
        tb_logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    log_dir = tb_logdir
    file_writer = tf.summary.create_file_writer(tb_logdir + "/custom_evaluation")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(tb_logdir, histogram_freq=1)
    callback_list.append(tensorboard_callback)
    # print(f"\nTensorboard logs path: {tb_logdir}\n")
    tf.compat.v1.logging.info(f"Tensorboard logs path: {tb_logdir}")

    """CSV Logger Callback """
    assert os.path.exists(log_dir), "CSV log path doesn't exist"
    csv = os.path.join(log_dir, "logs.csv")
    csv_logger = CSVLogger(
        csv,
        append=True,
        separator=";",
    )
    # print(f"\nModel CSV logs path: {csv}\n")
    tf.compat.v1.logging.info(f"Model CSV logs path: {csv}")
    callback_list.append(csv_logger)

    """Reduce LR Callback """
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.2, patience=1, min_lr=0.000001
    )
    callback_list.append(reduce_lr_callback)

    """Checkpoint Callback """
    if params["save_weights"]:
        assert os.path.exists(cn.MODEL_PATH), "MODEL_PATH doesn't exist"
        checkpoint_path = os.path.join(
            cn.MODEL_PATH, (Path(log_dir).parent).name, Path(log_dir).name
        )
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        assert os.path.exists(checkpoint_path), "checkpoint_path doesn't exist"
        checkpoint_path = os.path.join(
            checkpoint_path,
            "weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
        )
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
            monitor="val_accuracy",
        )
        callback_list.append(cp_callback)
        # print(f"\nModel Checkpoint path: {checkpoint_path}\n")
        tf.compat.v1.logging.info(f"Model Checkpoint path: {checkpoint_path}")

    return callback_list, log_dir
