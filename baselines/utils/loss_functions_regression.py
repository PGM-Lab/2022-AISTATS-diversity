import tensorflow as tf
from tensorflow.keras.losses import MSE

from baselines.utils import varianceBound


def mse_variance(predictions):
    # Predictions shape (n_models, n_samples)
    # Variance = E_D [ Var_\rho predictions]
    var_rho = tf.math.reduce_variance(predictions, axis=0)
    return tf.reduce_mean(var_rho)


def PAC2B(predictions, mse, l2_loss, FLAGS, **kwargs):
    variance = mse_variance(predictions)
    return tf.reduce_mean(mse) - variance + l2_loss


def PAC2B_val(predictions, predictions_valid, mse, l2_loss, FLAGS, **kwargs):
    # variance from train and validation
    # note: concat by axis=1 (dim 0 is the ensemble)
    variance_valid = mse_variance(tf.concat([predictions, predictions_valid], axis=1))

    # loss function
    return tf.reduce_mean(mse) - variance_valid + l2_loss


def Ensemble_mse(predictions, y_true, l2_loss, **kwargs):
    mse = MSE(y_true, tf.reduce_mean(predictions))
    return mse + l2_loss


def PACB(mse, l2_loss, **kwargs):
    return tf.reduce_mean(mse) + l2_loss


def compute_loss(loss_fn: str, **kwargs):

    if loss_fn == "PAC2B":
        loss = PAC2B(**kwargs)
    elif loss_fn == "PAC2B-val":
        loss = PAC2B_val(**kwargs)
    elif loss_fn == "Ensemble-MSE":
        loss = Ensemble_mse(**kwargs)
    elif loss_fn == "PACB":
        loss = PACB(**kwargs)
    else:
        raise ValueError("Wrong loss_fn code")

    return loss
