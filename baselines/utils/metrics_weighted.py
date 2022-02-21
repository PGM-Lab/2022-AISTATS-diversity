import itertools

import numpy as np
import tensorflow as tf
import uncertainty_metrics as um
from baselines.utils.metrics import tandem_loss, supervised_disagreement, \
    logLike_Variance, update_metrics_perModel, update_PACBayes_metrics

def update_metrics_weighted(metrics, labels, logits, num_models, weights = None):

    # Compute probs from logits
    probs = tf.nn.softmax(logits)

    # Initialice uniform weights if needed.
    #  otherwise, weights should have shape (num_models, 1, 1).
    #  this can be forced using tf.expand_dims twice
    if weights is None:
        weights = tf.constant(1/num_models * np.ones(num_models), 
                              dtype=np.float32, shape = (num_models,1,1)
                             )
    
    # Assert weights shapes for debugging
    # tf.debugging.assert_shapes([weights, (num_models, 1, 1)])

    # In this two metric, averaging is substituted by a summation given that
    # the probabilities are already weighted
    metrics['ensemble_accuracy'].update_state(labels,tf.reduce_sum(weights*probs, axis=0))
    metrics['ensemble_ece'].update_state(labels,tf.reduce_sum(weights*probs, axis=0))
    
    # Gibbs error is the same using weighted or not as it uses argmax
    metrics['gibbs_error'].update_state(
        gibbs_error_weighted(labels, probs, num_models, weights = weights[:,0,0])
    )
    metrics['ensemble_error'].update_state(ensemble_error_weighted(labels, probs, weights))

    metrics['gibbs_ce'].update_state(gibbs_cross_entropy(labels,logits, weights))
    metrics['ensemble_ce'].update_state(ensemble_cross_entropy(labels,logits, weights))

    metrics['variance'].update_state(variance_bound_weighted(labels,probs,weights))

    metrics['variance_T'].update_state(logLike_Variance(labels, probs, weights))

    metrics['data_size'].update_state(logits.shape[1])

    update_pairwise_diversity_weighted(metrics, labels, probs, num_models, weights)

    update_metrics_perModel(metrics, labels, logits, num_models)

    #update_PACBayes_metrics(metrics,num_models)


def variance_bound_weighted(labels, probs, weights):
    """Weighted empirical upper bound on the variance for an ensemble model.

    This term was introduced in arxiv.org/abs/1912.08335 to obtain a tighter
    PAC-Bayes upper bound; we use the empirical variance of Theorem 4.

    Args:
        probs: 
            tf.Tensor of shape [num_models, batch_size, n_classes].
        labels: 
            tf.Tensor of sparse labels, of shape [batch_size].
        weights:             
            tf.Tensor of shape [ensemble_size]. If None, uniform weights are used.

    Returns:
        A (float) upper bound on the empirical ensemble variance.
    """
    num_models = probs.shape[0]
    batch_size = probs.shape[1]
    labels = tf.cast(labels, dtype=tf.int32)

    # batch_indices maps point `i` to its associated label `l_i`.
    batch_indices = tf.stack([tf.range(batch_size), labels], axis=1)
    # Shape: [num_models, batch_size, batch_size].
    batch_indices = batch_indices * tf.ones([num_models, 1, 1], dtype=tf.int32)

    # Replicate batch_indices across the `num_models` index.
    ensemble_indices = tf.reshape(tf.range(num_models), [num_models, 1, 1])
    ensemble_indices = ensemble_indices * tf.ones([1, batch_size, 1],
                                                dtype=tf.int32)
    # Shape: [num_models, batch_size, n_classes].
    indices = tf.concat([ensemble_indices, batch_indices], axis=-1)

    # Shape: [n_models, n_points].
    # per_model_probs[n, i] contains the probability according to model `n` that
    # point `i` in the batch has its true label.
    per_model_probs = tf.gather_nd(probs, indices)

    max_probs = tf.reduce_max(per_model_probs, axis=0)  # Shape: [n_points]
    avg_probs = tf.reduce_sum(weights[:,0] * per_model_probs, axis=0)  # Shape: [n_points]

    # return .5 * tf.reduce_sum(
    #     weights * tf.square((per_model_probs - avg_probs) / max_probs)
    
    return .5 * tf.reduce_mean(
        tf.reduce_sum( weights[:,0]* tf.square((per_model_probs - avg_probs) / max_probs), axis = 0)
    )


def logLike_Variance(labels, probs, weights):
    """
    Computes the weighted loglikelihood variance.

    Args:
        labels: 
            tf.Tensor of shape [...].
        probs: 
            tf.Tensor of shape [ensemble_size, ..., num_classes].
        weights:
            tf.Tensor of shape [ensemble_size]. If None, uniform weights are used.
    Returns:
        tf.Tensor of shape [...].

    """
    ensemble_size = probs.shape[0]

    labels_ext = tf.cast(
        tf.reshape(tf.tile(labels, [ensemble_size]), [ensemble_size, labels.shape[0]]), tf.int32
        )
    # Compute log likelihood using categorical cross entropy error
    log_likelihood = -tf.keras.losses.sparse_categorical_crossentropy(labels_ext, probs)

    ## VarianceBoundTight
    probs = tf.math.exp(log_likelihood)
    mean, variance = tf.nn.weighted_moments(probs, axes = 0, frequency_weights = weights[:,0])
    
    # hfunction
    max = tf.reduce_max(probs, axis=0)
    inc = tf.math.log(mean) - tf.math.log(max)

    inc = tf.clip_by_value(inc, clip_value_min=-10., clip_value_max=-0.01)
    hfunction =  tf.math.divide(
        tf.math.divide(inc, tf.square(1 - tf.math.exp(inc))) + 1. / tf.multiply(tf.math.exp(inc), 
        1 - tf.math.exp(inc)),
        tf.square(max))

    variance = tf.reduce_mean(variance * \
        tf.stop_gradient(hfunction))
    return variance



def gibbs_error_weighted(labels, probs, num_models, aggregate=True, weights = None):
    """
    Computes the weighted Gibbs error.

    Args:
        labels: 
            tf.Tensor of shape [...].
        probs: 
            tf.Tensor of shape [ensemble_size, ..., num_classes].
        aggregate: 
            bool.
            Whether or not to average over the batch.
        weights:
            tf.Tensor of shape [ensemble_size]. If None, uniform weights are used.
    Returns:
        tf.Tensor of shape [...].

    """
    # Reshape labels and cast them to int32
    labels_ext = tf.cast(
        tf.reshape(tf.tile(labels, [num_models]), [num_models, labels.shape[0]]), tf.int32)
    
    # Most probable class for each model and sample 
    pred_ext = tf.argmax(probs, axis=-1, output_type=tf.int32)

    # Compute error mask
    result = tf.cast(pred_ext != labels_ext, tf.float32)

    if aggregate: # Average over batch/samples
        result = tf.reduce_mean(result, axis = 1)

    if weights is None:
        result = tf.reduce_mean(result, axis = 0)
    else: # Weighted average over ensemble models
        result = tf.reduce_sum(weights * result, axis = 0)

    return result

def ensemble_error_weighted(labels, probs, weights = None):
    """
    Computes the weighted ensemble error.

    Args:
        labels: 
            tf.Tensor of shape [num_samples].
        probs: 
            tf.Tensor of shape [ensemble_size, num_samples, num_classes].
        weights:
            tf.Tensor of shape [ensemble_size]. If Nome, uniform weights are used.
    Returns:
        tf.Tensor of shape [...].

    """
    if weights is None:
        probs = tf.reduce_mean(probs, axis=0)
    else: #Weighted average of probabilities
        probs = tf.reduce_sum(weights*probs, axis=0)
    
    # Get most probable prediction
    pred = tf.argmax(probs, axis=-1, output_type=tf.int32)

    # Compute mean error rate
    result = tf.reduce_mean(tf.cast(pred != tf.cast(labels, tf.int32), tf.float32)) 

    return result 

def gibbs_cross_entropy(labels, logits, weights = None, binary=False, aggregate=True):
    """Average cross entropy for ensemble members (Gibbs cross entropy).

    For each datapoint (x,y), the ensemble's Gibbs cross entropy is:

    ```
      - (1/ensemble_size) sum_{m=1}^ensemble_size log p(y|x,theta_m).
    ```
    Therefore, the weighted version is
    ```
      - sum_{m=1}^ensemble_size w_m log p(y|x,theta_m).
    ```

    The Gibbs cross entropy approximates the average cross entropy of a single
    model drawn from the (Gibbs) ensemble.

    Args:
        labels: 
            tf.Tensor of shape [...].
        logits: 
            tf.Tensor of shape [ensemble_size, ..., num_classes].
        weights:
            tf.Tensor of shape [ensemble_size]. If Nome, uniform weights are used
        binary: 
            bool. 
            Whether it is a binary classification (sigmoid as activation).
        aggregate: 
            bool. 
            Whether or not to average over the batch.

    Returns:
        tf.Tensor of shape [...].
    """
    # Logits need to be a tensor for the upcoming operations
    logits = tf.convert_to_tensor(logits)
    labels = tf.cast(labels, tf.int32)

    # Use sigmoid of softmax depending on if problem is binary
    if binary:
        ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)),
            logits=logits
        )
    else:
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)[:-1]),
            logits=logits
        )
    # Averaging over batch/samples
    if aggregate:
        ce = tf.reduce_mean(ce, axis = 1)

    if weights is None:
        nll = tf.reduce_mean(ce, axis = 0)
    else: # Weighted averaging over ensemble models. Weights has more dimensions than needed
        nll = tf.reduce_sum(weights[:,0]*ce, axis = 0)
        
    return nll



def ensemble_cross_entropy(labels, logits, weights, binary=False, aggregate=True):
    """Cross-entropy of an ensemble distribution.

    For each datapoint (x,y), the ensemble's negative log-probability is:

    ```
        -log p(y|x) = -log sum_{m=1}^{ensemble_size} exp(log p(y|x,theta_m)) +
                  log ensemble_size.
    ```
    Therefore, the weighted value is
    ```
        -log p(y|x) = -log sum_{m=1}^{ensemble_size} w_i exp(log p(y|x,theta_m)) =
            -log sum_{m=1}^{ensemble_size} exp(log w_i + log p(y|x,theta_m))
    ```

    The cross entropy is the expected negative log-probability with respect to
    the true data distribution.

    Args:
        labels: 
            tf.Tensor of shape [...].
        logits: 
            tf.Tensor of shape [ensemble_size, ..., num_classes].
        weights:
            tf.Tensor of shape [ensemble_size]. If Nome, uniform weights are use
        binary: 
            bool
            Whether it is a binary classification (sigmoid as activation).
        aggregate: 
            bool 
            Whether or not to average over the batch.

    Returns:
        tf.Tensor of shape [...].
    """
    # Logits must be a tensor for the upcoming functions
    logits = tf.convert_to_tensor(logits)
    labels = tf.cast(labels, tf.int32)
    
    if binary: # Use sigmoid of softmax depending on if problem is binary
        ce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)),
            logits=logits
        )
    else:
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)[:-1]),
            logits=logits
        )

    if weights is None:
        ensemble_size = float(logits.shape[0])
        nll = -tf.reduce_logsumexp(-ce, axis = 0) + tf.math.log(ensemble_size)
    else: # Weighted averaging over ensemble models
        nll = -tf.reduce_logsumexp(-ce + tf.math.log(weights[:,0]), axis = 0)

    # Averaging over batch/samples
    if aggregate:
        nll = tf.reduce_mean(nll)
    
    return nll



def update_pairwise_diversity_weighted(metrics, labels, probs, num_models, weights, error=None):
    """Average pairwise distance computation across models."""
    if probs.shape[0] != num_models:
        raise ValueError('The number of models {0} does not match '
                         'the probs length {1}'.format(num_models, probs.shape[0]))

    ### Metrics for combinations
    pairwise_weights = []
    pairwise_disagreement = []
    pairwise_kl_divergence = []
    pairwise_cosine_distance = []

    for pair in list(itertools.combinations(range(num_models), 2)):
        probs_1 = probs[pair[0]]
        probs_2 = probs[pair[1]]

        pair_weight = weights[pair[0]] * weights[pair[1]]
        pairwise_weights.append(pair_weight)

        pairwise_kl_divergence.append(
            tf.reduce_mean(um.kl_divergence(probs_1, probs_2))
        )
        pairwise_disagreement.append(
            pair_weight*um.disagreement(probs_1, probs_2)
        )
        pairwise_cosine_distance.append(
            pair_weight*um.cosine_distance(probs_1, probs_2)
        )
    # Re-escaling needed.
    total_weight = tf.reduce_sum(tf.stack(pairwise_weights))

    average_cosine_distance = tf.reduce_sum(tf.stack(pairwise_cosine_distance))/total_weight
    average_disagreement = tf.reduce_sum(tf.stack(pairwise_disagreement))/total_weight
    average_kl_divergence = tf.reduce_mean(tf.stack(pairwise_kl_divergence))/total_weight
    if error is not None:
        average_disagreement /= (error + tf.keras.backend.epsilon())

    ### Metrics for product
    pairwise_tandom_loss = []
    pairwise_supervised_disagreement = []
    for pair in list(itertools.product(range(num_models), repeat = 2)):
        probs_1 = probs[pair[0]]
        probs_2 = probs[pair[1]]

        pair_weight = weights[pair[0]] * weights[pair[1]]

        pairwise_tandom_loss.append(
            pair_weight*tandem_loss(labels, probs_1, probs_2)
        )
        pairwise_supervised_disagreement.append(
            pair_weight*supervised_disagreement(labels, probs_1, probs_2)
        )

    average_tandem_loss = tf.reduce_sum(tf.stack(pairwise_tandom_loss))
    average_supervised_disagreement = tf.reduce_sum(tf.stack(pairwise_supervised_disagreement))


    metrics['disagreement'].update_state(average_disagreement)
    metrics['average_kl'].update_state(average_kl_divergence)
    metrics['cosine_similarity'].update_state(average_cosine_distance)
    metrics['tandem_loss'].update_state(average_tandem_loss)
    metrics['supervised_disagreement'].update_state(average_supervised_disagreement)


def compare_metrics(metrics1, metrics2):
    """
    Compares and checks if the given two metrics are equal. Made for debugging purposes.
    """
    for k in metrics1.keys():
        n1 = metrics1[k].result().numpy()
        n2 = metrics2[k].result().numpy()
        if np.isclose(n1, n2):
            print("\t - ", k, "Correct")
        else:
            print("\t - ", k, "Error")
            print("\t\t", n1, " != ", n2)



def logVarianceBoundPAC2B1_weighted(log_likelihood, effectiveNumModel, weights):

    constant2 = tf.keras.backend.constant(2.)

    effectiveNumModel_float = tf.keras.backend.cast_to_floatx(effectiveNumModel)

    logmax = tf.stop_gradient(tf.reduce_max(log_likelihood, axis=0))
    logmean = tf.reduce_logsumexp(log_likelihood + tf.math.log(weights[:,0]), axis=0)
    inc = logmean - logmax
    inc = tf.clip_by_value(inc, clip_value_min=-10., clip_value_max=-0.01)

    hmax = tf.stop_gradient(
        inc / tf.math.pow(1 - tf.math.exp(inc), constant2) + tf.math.pow(tf.math.exp(inc) * (1 - tf.math.exp(inc)), -1))

    variance = 0.
    for i in range(effectiveNumModel):
        variance += weights[i,0,0] * tf.exp(constant2 * log_likelihood[i] - constant2 * logmax)
        for j in range(effectiveNumModel):
            variance -= weights[i,0,0]*weights[j,0,0]\
                * tf.exp(log_likelihood[i] + log_likelihood[j] - constant2 * logmax)

    variance *= hmax
    # Mean over points
    variance = tf.reduce_mean(variance)
    return variance

