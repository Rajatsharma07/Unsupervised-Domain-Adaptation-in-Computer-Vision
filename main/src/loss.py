import tensorflow as tf


def kl_divergence(model, source_output, target_output, percent_lambda):
    kl_loss = -0.5 * tf.reduce_mean(
        target_output - tf.square(source_output) - tf.exp(target_output) + 1
    )
    kl_loss = percent_lambda * kl_loss
    model.add_loss(kl_loss)
    return kl_loss


def CORAL(source_output, target_output, percent_lambda=0.5):

    source_batch_size = tf.cast(tf.shape(source_output)[0], tf.float64)
    target_batch_size = tf.cast(tf.shape(target_output)[0], tf.float64)
    d = tf.cast(tf.shape(source_output)[1], tf.float64)

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


def coral_loss(source_output, target_output, percent_lambda=0.5):
    h_src = source_output
    h_trg = target_output
    # regularized covariances (D-Coral is not regularized actually..)
    # First: subtract the mean from the data matrix
    batch_size = tf.cast(tf.shape(h_src)[0], tf.float32)
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    cov_source = (1.0 / (batch_size - 1)) * tf.matmul(
        h_src, h_src, transpose_a=True
    )  # + gamma * tf.eye(self.hidden_repr_size)
    cov_target = (1.0 / (batch_size - 1)) * tf.matmul(
        h_trg, h_trg, transpose_a=True
    )  # + gamma * tf.eye(self.hidden_repr_size)
    return percent_lambda * tf.reduce_mean(
        tf.square(tf.subtract(cov_source, cov_target))
    )


# calculate LogCoral loss
def log_coral_loss(source_output, target_output, gamma=1e-3, percent_lambda=0.5):
    # regularized covariances result in inf or nan
    # First: subtract the mean from the data matrix
    h_src = source_output
    h_trg = target_output
    batch_size = tf.cast(tf.shape(h_src)[0], tf.float32)
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    cov_source = (1.0 / (batch_size - 1)) * tf.matmul(
        h_src, h_src, transpose_a=True
    )  # + gamma * tf.eye(self.hidden_repr_size)
    cov_target = (1.0 / (batch_size - 1)) * tf.matmul(h_trg, h_trg, transpose_a=True)
    eig_source = tf.self_adjoint_eig(cov_source)
    eig_target = tf.self_adjoint_eig(cov_target)
    log_cov_source = tf.matmul(
        eig_source[1],
        tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True),
    )
    log_cov_target = tf.matmul(
        eig_target[1],
        tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True),
    )
    return percent_lambda * tf.reduce_mean(
        tf.square(tf.subtract(log_cov_source, log_cov_target))
    )


def SteinCoral_loss(source_output, target_output, percent_lambda=0.5):
    h_src = source_output
    h_trg = target_output
    batch_size = tf.cast(tf.shape(h_src)[0], tf.float32)
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    cov_source = (1.0 / (batch_size - 1)) * tf.matmul(
        h_src, h_src, transpose_a=True
    ) + 5.0 * tf.eye(int(h_src.shape[1]))
    cov_target = (1.0 / (batch_size - 1)) * tf.matmul(
        h_trg, h_trg, transpose_a=True
    ) + 5.0 * tf.eye(int(h_trg.shape[1]))
    loss = tf.linalg.logdet(0.5 * (cov_source + cov_target)) - 0.5 * tf.linalg.logdet(
        tf.matmul(cov_source, cov_target)
    )
    return percent_lambda * loss


def JeffCoral_loss(source_output, target_output, percent_lambda=0.5):
    h_src = source_output
    h_trg = target_output
    batch_size = tf.cast(tf.shape(h_src)[0], tf.float32)
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    cov_source = (1.0 / (batch_size - 1)) * tf.matmul(
        h_src, h_src, transpose_a=True
    ) + 5.0 * tf.eye(int(h_src.shape[1]))
    cov_target = (1.0 / (batch_size - 1)) * tf.matmul(
        h_trg, h_trg, transpose_a=True
    ) + 5.0 * tf.eye(int(h_trg.shape[1]))
    loss = 0.5 * tf.trace(
        tf.matmul(cov_source, tf.linalg.inv(cov_target))
        + tf.matmul(cov_target, tf.linalg.inv(cov_source))
    ) - tf.cast(h_src.shape[1], tf.float32)
    return percent_lambda * tf.abs(loss)
