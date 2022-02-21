import itertools

import math
import numpy as np
import tensorflow as tf
import uncertainty_metrics as um
from baselines.utils.loss_functions_regression import mse_variance

MSE = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )

def createMetrics(num_models):
    metrics = {
        "ensemble_mse": tf.keras.metrics.MeanSquaredError(),
        "gibbs_mse": tf.keras.metrics.Mean(),
        "variance": tf.keras.metrics.Mean(),
    }

    for index in range(num_models):
        metrics.update(createMetricsPerModel(index))

    return metrics


def createMetricsPerModel(index):
    metrics = {
        "model_mse_" + str(index): tf.keras.metrics.MeanSquaredError(),
    }
    return metrics


def getListOfMetrics(metrics):
    return [measures.replace("/", "_").split("_")[1] for measures in metrics.keys()]


def update_metrics(metrics, y_true, y_preds, num_models):

    metrics["ensemble_mse"].update_state(
        y_true=y_true, y_pred=tf.reduce_mean(y_preds, axis=0)
    )
    
    if len(y_true.shape) == 1:
        tiled_y_true = tf.reshape(tf.tile(y_true, [num_models]), [num_models, -1, 1])
    else:
        tiled_y_true = tf.reshape(
            tf.tile(y_true, [num_models, 1, 1]),
            [*y_preds.shape],
        )
    # shape [n_models, batch_size, ...]
    mse = MSE(
        y_true = tiled_y_true,
        y_pred = y_preds
    )
    # If target images are multidimensional (images)
    #  reduce over those dimensions
    if len(mse.shape) > 2:
        mse = tf.reduce_mean(mse, axis = [np.arange(2, len(mse.shape) + 1)])
    
    # Reduce over batch_size
    mse = tf.reduce_mean(mse, axis = -1)               
    
    metrics["gibbs_mse"].update_state(mse)
    
    metrics["variance"].update_state(mse_variance(y_preds))

    update_metrics_perModel(metrics, y_true, y_preds, num_models)


def update_metrics_perModel(metrics, y_true, y_preds, num_models):

    for index in range(num_models):
        metrics["model_mse_" + str(index)].update_state(y_true, y_preds[index, :])
