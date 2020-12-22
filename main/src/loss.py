import tensorflow as tf


def kl_divergence(model, source_output, target_output, percent_lambda):
    kl_loss = -0.5 * tf.reduce_mean(
        target_output - tf.square(source_output) - tf.exp(target_output) + 1
    )
    kl_loss = percent_lambda * kl_loss
    model.add_loss(kl_loss)
    return kl_loss
