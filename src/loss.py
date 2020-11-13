import tensorflow as tf


def coral_loss(model, source_output, target_output, percent_lambda):

    source_batch_size = tf.cast(tf.shape(source_output)[0], tf.float32)
    target_batch_size = tf.cast(tf.shape(target_output)[0], tf.float32)
    d = tf.cast(tf.shape(source_output)[1], tf.float32)

    # Source covariance
    xm = source_output - tf.reduce_mean(source_output, 0, keepdims=True)
    xc = tf.matmul(tf.transpose(xm), xm) / source_batch_size

    # Target covariance
    xmt = target_output - tf.reduce_mean(target_output, 0, keepdims=True)
    xct = tf.matmul(tf.transpose(xmt), xmt) / target_batch_size

    loss = tf.reduce_sum(tf.multiply((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    model.add_loss(percent_lambda * loss)


def kl_divergence(model, source_output, target_output, percent_lambda):
    kl_loss = -0.5 * tf.reduce_mean(
        target_output.output
        - tf.square(source_output.output)
        - tf.exp(target_output.output)
        + 1
    )
    model.add_loss(kl_loss)
