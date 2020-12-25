import tensorflow as tf


def kl_divergence(model, source_output, target_output, percent_lambda):
    kl_loss = -0.5 * tf.reduce_mean(
        target_output - tf.square(source_output) - tf.exp(target_output) + 1
    )
    kl_loss = percent_lambda * kl_loss
    model.add_loss(kl_loss)
    return kl_loss


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
