
import tensorflow as tf

from baselines.utils import varianceBound


def PAC2B(log_likelihood, l2_loss, FLAGS, **kwargs):
    variance = varianceBound.logVarianceBoundPAC2B1(log_likelihood, FLAGS.ensemble_size)
    return -tf.reduce_mean(log_likelihood) - variance + l2_loss


def PAC2B_val(log_likelihood, log_likelihood_valid, l2_loss, FLAGS, **kwargs):

    # ll = [n_models, batch_size]
    # ll_valid = [n_models, batch_size
    # variance from train and validation
    # note: concat by axis=1 (dim 0 is the ensemble)
    variance_valid = varianceBound.logVarianceBoundPAC2B1(
        tf.concat([log_likelihood, log_likelihood_valid], axis=1), FLAGS.ensemble_size
    )

    # loss function
    return -tf.reduce_mean(log_likelihood) - variance_valid + l2_loss


def PAC2B_val_unsupervised(
    log_likelihood, log_likelihood_valid, l2_loss, FLAGS, **kwargs
):

    # variance from train and validation
    # note: concat by axis=1 (dim 0 is the ensemble)
    variance_valid = varianceBound.logVarianceBoundPAC2B1(
        tf.concat([log_likelihood, log_likelihood_valid], axis=1), FLAGS.ensemble_size
    )

    # loss function
    return -tf.reduce_mean(log_likelihood) - FLAGS.beta * variance_valid + l2_loss


def PAC2B_val_unsupervised2(
    probs_valid, log_likelihood, log_likelihood_valid, l2_loss, FLAGS, **kwargs
):
    """
    Computes PAC2B loss using unsupervised validation data.
    Arguments:
        probs_valid: tf.Tensor of shape [batch_size, n_classes]
        log_likelihood: tf.Tensor of shape [n_models, batch_size]
        log_likelihood_valid: tf.Tensor of shape [n_models, batch_size, n_classes]
        l2_loss: float value
    """

    # ll = [n_models, batch_size]
    # ll_valid = [n_models, batch_size, n_classes]
    n_classes = probs_valid.shape[-1]

    # [batch_size_train]
    variance_train = varianceBound.logVarianceBoundPAC2B1(log_likelihood, 
                                                          FLAGS.ensemble_size,
                                                          reduce_mean = False)
    # [n_classes, batch_size_val]
    variance_valid = tf.convert_to_tensor([
            probs_valid[:, i]
            * varianceBound.logVarianceBoundPAC2B1(log_likelihood_valid[i],
                FLAGS.ensemble_size,
                reduce_mean=False,
            )
        for i in range(n_classes)
    ])
    # [batch_size_valid]
    variance_valid = tf.reduce_sum(variance_valid, axis = 0)
    variance = tf.reduce_mean(tf.concat([variance_train, variance_valid], axis = 0))

    # loss function
    return -tf.reduce_mean(log_likelihood) - FLAGS.beta * variance + l2_loss


def Ensemble_CE(logits, labels, l2_loss, **kwargs):
    probs = tf.nn.softmax(logits)
    probs = tf.reduce_mean(probs, axis=0)
    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
    )
    return negative_log_likelihood + l2_loss


def PACB(log_likelihood, l2_loss, **kwargs):
    return -tf.reduce_mean(log_likelihood) + l2_loss


def compute_l2_loss(model, l2penalty):
    filtered_variables = []
    for var in model.trainable_variables:
        # Apply l2 on the BN parameters and bias terms.
        if "kernel" in var.name or "batch_norm" in var.name or "bias" in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))
    l2_loss = l2penalty * tf.nn.l2_loss(tf.concat(filtered_variables, axis=0))
    return l2_loss


def compute_loss(loss_fn: str, **kwargs):

    if loss_fn == "PAC2B":
        loss = PAC2B(**kwargs)
    elif loss_fn == "PAC2B-val":
        loss = PAC2B_val(**kwargs)
    elif loss_fn == "PAC2B-val-unsupervised":
        loss = PAC2B_val_unsupervised(**kwargs)
    elif loss_fn == "PAC2B-val-unsupervised2":
        loss = PAC2B_val_unsupervised2(**kwargs)
    elif loss_fn == "Ensemble-CE":
        loss = Ensemble_CE(**kwargs)
    elif loss_fn == "PACB":
        loss = PACB(**kwargs)
    else:
        raise ValueError("Wrong loss_fn code")
        # loss = None

    return loss
