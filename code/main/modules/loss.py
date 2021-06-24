import tensorflow as tf


def kl_divergence(source_output, target_output, percent_lambda):
    """[KL Divergence loss function]

    Args:
        source_output ([tf tensor]): [Source feature map tensor]
        target_output ([tf tensor]): [Target feature map tensor]
        percent_lambda (weighting factor, optional): [KL loss weighting factor]. Defaults to 0.5.

    Returns:
        [tf tensor]: [KL loss per batch]
    """
    kl_loss = -0.5 * tf.reduce_mean(
        target_output - tf.square(source_output) - tf.exp(target_output) + 1
    )
    kl_loss = percent_lambda * kl_loss
    return kl_loss


def CORAL(source_output, target_output, percent_lambda=0.5):
    """[Deep CORAL loss function]

    Args:
        source_output ([tf tensor]): [Source feature map tensor]
        target_output ([tf tensor]): [Target feature map tensor]
        percent_lambda (weighting factor, optional): [CORAL loss weighting factor]. Defaults to 0.5.

    Returns:
        [tf tensor]: [CORAL loss per batch]
    """
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
    # loss = tf.sqrt(tf.reduce_sum(tf.multiply((xc - xct), (xc - xct))))
    loss = tf.reduce_sum(tf.multiply((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    loss = percent_lambda * loss
    return loss


def log_coral_loss(source_output, target_output, percent_lambda=0.5):
    """[Calculate LogCoral loss]

       Args:
        source_output ([tf tensor]): [Source feature map tensor]
        target_output ([tf tensor]): [Target feature map tensor]
        percent_lambda (weighting factor, optional): [Log CORAL loss weighting factor]. Defaults to 0.5.

    Returns:
        [tf tensor]: [Log CORAL loss per batch]
    """
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
