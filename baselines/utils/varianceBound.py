import numpy as np
import tensorflow as tf
import itertools
import edward2 as ed

constant2 = tf.keras.backend.constant(2.)

def hfunction(probs):
    mean = tf.reduce_mean(probs,axis=0)
    max = tf.reduce_max(probs, axis=0)
    inc = tf.math.log(mean) - tf.math.log(max)
    inc = tf.clip_by_value(inc, clip_value_min=-10., clip_value_max=-0.01)
    return tf.math.divide(
        tf.math.divide(inc, tf.square(1 - tf.math.exp(inc))) + 1. / tf.multiply(tf.math.exp(inc), 1 - tf.math.exp(inc)),
        tf.square(max))

def varianceBoundTight(log_likelihood):
    probs = tf.math.exp(log_likelihood)
    variance = tf.reduce_mean(tf.math.reduce_variance(probs,axis=0)*tf.stop_gradient(hfunction(probs)))
    return variance

def varianceBound(probs):
    max = tf.reduce_max(probs, axis=0)
    return tf.reduce_mean(tf.divide(tf.math.reduce_variance(probs,axis=0),constant2*max*max))

def logVarianceBoundPAC2B0(log_likelihood):
    log_likelihood1 = tf.expand_dims(log_likelihood[0], 1)
    log_likelihood2 = tf.expand_dims(log_likelihood[1], 1)

    logmax = tf.stop_gradient(tf.math.maximum(log_likelihood1, log_likelihood2))
    logmean_logmax = tf.math.reduce_logsumexp(tf.concat([log_likelihood1 - logmax, log_likelihood2 - logmax], 1),
                                              axis=1) - tf.math.log(constant2)

    alpha = tf.expand_dims(logmean_logmax, 1)
    alpha = tf.clip_by_value(alpha, clip_value_min=-10., clip_value_max=-0.01)

    hmax = tf.stop_gradient(
        alpha / tf.math.pow(1 - tf.math.exp(alpha), constant2) + tf.math.pow(tf.math.exp(alpha) * (1 - tf.math.exp(alpha)), -1))

    variance = (tf.reduce_mean(tf.exp(constant2 * log_likelihood1 - constant2 * logmax) * hmax) - tf.reduce_mean(
        tf.exp(log_likelihood1 + log_likelihood2 - constant2 * logmax) * hmax))
    variance = variance + (tf.reduce_mean(tf.exp(constant2 * log_likelihood2 - constant2 * logmax) * hmax) - tf.reduce_mean(
        tf.exp(log_likelihood1 + log_likelihood2 - constant2 * logmax) * hmax))
    variance = variance / constant2

    log_likelihood = 0.5 * tf.reduce_mean(log_likelihood1) + 0.5 * tf.reduce_mean(log_likelihood2)

    return log_likelihood, variance

def logVarianceBoundPAC2B1ForLoop(log_likelihood):
    effectiveNumModel = tf.shape(log_likelihood)[0]
    effectiveNumModel_float = tf.keras.backend.cast_to_floatx(effectiveNumModel)

    logmax = tf.stop_gradient(tf.reduce_max(log_likelihood, axis=0))
    logmean = tf.reduce_logsumexp(log_likelihood, axis=0) - tf.math.log(effectiveNumModel_float)
    inc = logmean - logmax
    inc = tf.clip_by_value(inc, clip_value_min=-10., clip_value_max=-0.01)

    hmax = tf.stop_gradient(
        inc / tf.math.pow(1 - tf.math.exp(inc), constant2) + tf.math.pow(tf.math.exp(inc) * (1 - tf.math.exp(inc)), -1))

    variance = 0.
    for i in range(effectiveNumModel):
        variance += tf.exp(constant2 * log_likelihood[i] - constant2 * logmax) / effectiveNumModel_float
        for j in range(effectiveNumModel):
            variance -= tf.exp(log_likelihood[i] + log_likelihood[j] - constant2 * logmax) / (
                    effectiveNumModel_float * effectiveNumModel_float)

    variance *= hmax
    log_likelihood = tf.reduce_mean(log_likelihood)
    variance = tf.reduce_mean(variance)
    return log_likelihood, variance


def logVarianceBoundPAC2B1(log_likelihood, effectiveNumModel, reduce_mean = True):
    effectiveNumModel_float = tf.keras.backend.cast_to_floatx(effectiveNumModel)

    logmax = tf.stop_gradient(tf.reduce_max(log_likelihood, axis=0))
    logmean = tf.reduce_logsumexp(log_likelihood, axis=0) - tf.math.log(effectiveNumModel_float)
    inc = logmean - logmax
    inc = tf.clip_by_value(inc, clip_value_min=-10., clip_value_max=-0.01)

    hmax = tf.stop_gradient(
        inc / tf.math.pow(1 - tf.math.exp(inc), constant2) + tf.math.pow(tf.math.exp(inc) * (1 - tf.math.exp(inc)), -1))

    variance = tf.reduce_mean(tf.exp(constant2 * log_likelihood - constant2 * logmax),axis=0)
    for j in range(effectiveNumModel):
        variance -= tf.reduce_mean(tf.exp(log_likelihood + log_likelihood[j] - constant2 * logmax),axis=0) / (
                effectiveNumModel_float)

    variance *= hmax
    if reduce_mean:
        variance = tf.reduce_mean(variance)
    return variance
