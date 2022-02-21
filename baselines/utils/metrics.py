import itertools

import math
import numpy as np
import tensorflow as tf
import uncertainty_metrics as um
from baselines.utils import varianceBound


def createMetrics(num_models):
    metrics = {
        'ensemble_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'ensemble_ece': um.ExpectedCalibrationError(num_bins=15),
        'gibbs_error': tf.keras.metrics.Mean(),
        'ensemble_error': tf.keras.metrics.Mean(),
        'gibbs_ce': tf.keras.metrics.Mean(),
        'ensemble_ce': tf.keras.metrics.Mean(),
        'variance': tf.keras.metrics.Mean(),
        'variance_T': tf.keras.metrics.Mean(),
        'data_size' : tf.keras.metrics.Sum()
    }

    create_PACBayes_metrics(metrics)

    create_pairwise_diversity_metrics(metrics)

    for index in range(num_models):
        metrics.update(createMetricsPerModel(index))

    return metrics

def createMetricsPerModel(index):
    metrics = {
        'model_ece_'+str(index): um.ExpectedCalibrationError(num_bins=15),
        'model_error_'+str(index): tf.keras.metrics.Mean(),
        'model_ce_'+str(index): tf.keras.metrics.Mean(),
    }
    return metrics


def getListOfMetrics(metrics):
    return [measures.replace('/','_').split('_')[1] for measures in metrics.keys()]


def update_metrics(metrics, labels, logits, num_models):
    probs = tf.nn.softmax(logits)

    metrics['ensemble_accuracy'].update_state(labels,tf.reduce_mean(probs, axis=0))
    metrics['ensemble_ece'].update_state(labels,tf.reduce_mean(probs, axis=0))

    metrics['gibbs_error'].update_state(gibbs_error(labels,probs,num_models))
    metrics['ensemble_error'].update_state(ensemble_error(labels,probs,num_models))

    metrics['gibbs_ce'].update_state(gibbs_cross_entropy(labels,logits))
    metrics['ensemble_ce'].update_state(um.ensemble_cross_entropy(labels,logits))

    metrics['variance'].update_state(um.variance_bound(probs,labels,num_models))

    variance = logLike_Variance(labels,probs,num_models)
    metrics['variance_T'].update_state(variance)

    metrics['data_size'].update_state(logits.shape[1])

    update_pairwise_diversity(metrics, labels, probs, num_models)

    update_metrics_perModel(metrics, labels, logits, num_models)

    update_PACBayes_metrics(metrics,num_models)


def update_metrics_perModel(metrics, labels, logits, num_models):
    probs = tf.nn.softmax(logits)
    error = gibbs_error(labels,probs,num_models, aggregate=False)
    ce = gibbs_cross_entropy(labels,logits,aggregateEnsemble=False)

    for index in range(num_models):
        metrics['model_ece_'+str(index)].update_state(labels,probs[index])
        metrics['model_error_'+str(index)].update_state(error[index])
        metrics['model_ce_'+str(index)].update_state(ce[index])

def create_pairwise_diversity_metrics(metrics):
    metrics['disagreement'] = tf.keras.metrics.Mean()
    metrics['average_kl'] = tf.keras.metrics.Mean()
    metrics['cosine_similarity'] = tf.keras.metrics.Mean()
    metrics['tandem_loss'] = tf.keras.metrics.Mean()
    metrics['supervised_disagreement'] = tf.keras.metrics.Mean()


def create_PACBayes_metrics(metrics):
    metrics['PACBayesGAP_ce'] = tf.keras.metrics.Mean()
    metrics['PAC2BayesGAP_ce'] = tf.keras.metrics.Mean()
    metrics['PAC2TBayesGAP_ce'] = tf.keras.metrics.Mean()

    metrics['PACBayesGAP_01'] = tf.keras.metrics.Mean()
    metrics['PAC2BayesGAP_01'] = tf.keras.metrics.Mean()


def update_PACBayes_metrics(metrics,num_models):
    metrics['PACBayesGAP_ce'].reset_states()
    metrics['PACBayesGAP_ce'].update_state(PACBayesGAP_ce(metrics,num_models))

    metrics['PAC2BayesGAP_ce'].reset_states()
    metrics['PAC2BayesGAP_ce'].update_state(PAC2BayesGAP_ce(metrics,num_models))

    metrics['PAC2TBayesGAP_ce'].reset_states()
    metrics['PAC2TBayesGAP_ce'].update_state(PAC2TBayesGAP_ce(metrics,num_models))

    metrics['PACBayesGAP_01'].reset_states()
    metrics['PACBayesGAP_01'].update_state(PACBayesGAP_01(metrics,num_models))

    metrics['PAC2BayesGAP_01'].reset_states()
    metrics['PAC2BayesGAP_01'].update_state(PAC2BayesGAP_01(metrics,num_models))


def PACBayesGAP_ce(metrics,num_models):
    data_size = metrics['data_size'].result()
    ce = []
    for index in range(num_models):
        ce.append(metrics['model_ce_'+str(index)].result())

    boundEnsemble = metrics['gibbs_ce'].result() + 0.0/data_size
    boundBestModel = tf.reduce_min(ce) + math.log(num_models) / data_size

    return  boundEnsemble - boundBestModel


def PAC2BayesGAP_ce(metrics,num_models):
    data_size = metrics['data_size'].result()

    ce = []
    for index in range(num_models):
        ce.append(metrics['model_ce_' + str(index)].result())

    boundEnsemble = metrics['gibbs_ce'].result()  - metrics['variance'].result() + 0.0 / data_size
    boundBestModel = tf.reduce_min(ce) + math.log(num_models) / data_size

    return boundEnsemble - boundBestModel


def PAC2TBayesGAP_ce(metrics,num_models):
    data_size = metrics['data_size'].result()

    ce = []
    for index in range(num_models):
        ce.append(metrics['model_ce_' + str(index)].result())

    boundEnsemble = metrics['gibbs_ce'].result()  - metrics['variance_T'].result() + 0.0 / data_size
    boundBestModel = tf.reduce_min(ce) + math.log(num_models) / data_size

    return boundEnsemble - boundBestModel

def PACBayesGAP_01(metrics,num_models):
    data_size = metrics['data_size'].result()

    error = []
    for index in range(num_models):
        error.append(metrics['model_error_' + str(index)].result())

    cte = math.log(2.) + 0.5*math.log(2.) - math.log(0.05)

    lambda_ensemble = 2*data_size*metrics['gibbs_error'].result()
    lambda_ensemble /= 0.0  + cte
    lambda_ensemble = tf.sqrt(lambda_ensemble+1) + 1
    lambda_ensemble = 2/lambda_ensemble
    boundEnsemble = metrics['gibbs_error'].result()/(1-lambda_ensemble/2.) + (0.0 + cte) / (data_size*lambda_ensemble*(1-lambda_ensemble/2.))

    best_error = tf.reduce_min(error)
    lambda_bestmodel = 2*data_size*best_error
    lambda_bestmodel /= math.log(num_models)  + cte
    lambda_bestmodel = tf.sqrt(lambda_bestmodel+1) + 1
    lambda_bestmodel = 2/lambda_bestmodel
    boundBestModel = best_error/(1-lambda_bestmodel/2.) + (math.log(num_models) + cte) / (data_size*lambda_bestmodel*(1-lambda_bestmodel/2.))

    return 2*boundEnsemble - 2*boundBestModel

def PAC2BayesGAP_01(metrics,num_models):
    data_size = metrics['data_size'].result()

    error = []
    for index in range(num_models):
        error.append(metrics['model_error_' + str(index)].result())

    cte = math.log(2.)  + 0.5*math.log(2.) - math.log(0.05)

    lambda_ensemble = 2*data_size*metrics['tandem_loss'].result()
    lambda_ensemble /= 2*0.0  + cte
    lambda_ensemble = tf.sqrt(lambda_ensemble+1) + 1.
    lambda_ensemble = 2./lambda_ensemble
    boundEnsemble = metrics['tandem_loss'].result()/(1-lambda_ensemble/2.) + (2*0.0 + cte) / (data_size*lambda_ensemble*(1-lambda_ensemble/2.))

    best_error = tf.reduce_min(error)
    lambda_bestmodel = 2.*data_size*best_error
    lambda_bestmodel /= 2.*math.log(num_models) + cte
    lambda_bestmodel = tf.sqrt(lambda_bestmodel+1) + 1.
    lambda_bestmodel = 2./lambda_bestmodel
    boundBestModel = best_error/(1-lambda_bestmodel/2.) + (2*math.log(num_models) + cte) / (data_size*lambda_bestmodel*(1-lambda_bestmodel/2.))

    return 4.*boundEnsemble - 4.*boundBestModel



def logLike_Variance(labels, probs, num_models):
    labels_ext = tf.cast(
        tf.reshape(tf.tile(labels, [num_models]), [num_models, labels.shape[0]]), tf.int32)
    log_likelihood = -tf.keras.losses.sparse_categorical_crossentropy(labels_ext,probs)
    return varianceBound.varianceBoundTight(log_likelihood)

def gibbs_error(labels, probs, num_models, aggregate=True):
    labels_ext = tf.cast(
        tf.reshape(tf.tile(labels, [num_models]), [num_models, labels.shape[0]]), tf.int32)
    pred_ext = tf.argmax(probs, axis=-1, output_type=tf.int32)

    result = tf.cast(pred_ext != labels_ext, tf.float32)

    if aggregate:
        result = tf.reduce_mean(result)

    return result

def ensemble_error(labels, probs, num_models):
    probs = tf.reduce_mean(probs, axis=0)
    pred = tf.argmax(probs, axis=-1, output_type=tf.int32)

    return  tf.reduce_mean(tf.cast(pred != tf.cast(labels, tf.int32), tf.float32))


def update_pairwise_diversity(metrics, labels, probs, num_models, error=None):
    """Average pairwise distance computation across models."""
    if probs.shape[0] != num_models:
        raise ValueError('The number of models {0} does not match '
                         'the probs length {1}'.format(num_models, probs.shape[0]))

    pairwise_disagreement = []
    pairwise_kl_divergence = []
    pairwise_cosine_distance = []
    pairwise_tandom_loss = []
    pairwise_supervised_disagreement = []
    
    for pair in list(itertools.combinations(range(num_models), 2)):
        probs_1 = probs[pair[0]]
        probs_2 = probs[pair[1]]
        pairwise_disagreement.append(um.disagreement(probs_1, probs_2))
        pairwise_kl_divergence.append(
            tf.reduce_mean(um.kl_divergence(probs_1, probs_2, clip=True)))
        pairwise_cosine_distance.append(um.cosine_distance(probs_1, probs_2))

    for pair in list(itertools.product(range(num_models), repeat = 2)):
        probs_1 = probs[pair[0]]
        probs_2 = probs[pair[1]]

        pairwise_tandom_loss.append(
            tandem_loss(labels, probs_1, probs_2)
        )
        pairwise_supervised_disagreement.append(
            supervised_disagreement(labels, probs_1, probs_2)
        )

    # TODO(ghassen): we could also return max and min pairwise metrics.
    average_disagreement = tf.reduce_mean(tf.stack(pairwise_disagreement))
    if error is not None:
        average_disagreement /= (error + tf.keras.backend.epsilon())
    average_kl_divergence = tf.reduce_mean(tf.stack(pairwise_kl_divergence))
    average_cosine_distance = tf.reduce_mean(tf.stack(pairwise_cosine_distance))
    average_tandem_loss = tf.reduce_mean(tf.stack(pairwise_tandom_loss))
    average_supervised_disagreement = tf.reduce_mean(tf.stack(pairwise_supervised_disagreement))

    metrics['disagreement'].update_state(average_disagreement)
    metrics['average_kl'].update_state(average_kl_divergence)
    metrics['cosine_similarity'].update_state(average_cosine_distance)
    metrics['tandem_loss'].update_state(average_tandem_loss)
    metrics['supervised_disagreement'].update_state(average_supervised_disagreement)


def tandem_loss(labels,logits_1, logits_2):
  """ Tandem loss """
  labels = tf.cast(labels, tf.int32)
  preds_1 = tf.argmax(logits_1, axis=-1, output_type=tf.int32)
  preds_2 = tf.argmax(logits_2, axis=-1, output_type=tf.int32)
  return tf.reduce_mean(tf.cast(tf.math.logical_and(preds_1 != labels,preds_2 != labels), tf.float32))

def supervised_disagreement(labels, logits_1, logits_2):
  """ Supervised disagreement """
  labels = tf.cast(labels, tf.int32)
  preds_1 = tf.argmax(logits_1, axis=-1, output_type=tf.int32)
  preds_2 = tf.argmax(logits_2, axis=-1, output_type=tf.int32)
  par1 = tf.reduce_mean(tf.cast(tf.math.logical_and(preds_1 == labels,preds_2 != labels), tf.float32))
  par2 = tf.reduce_mean(tf.cast(tf.math.logical_and(preds_2 == labels,preds_1 != labels), tf.float32))
  return 0.5*par1 + 0.5*par2




def gibbs_cross_entropy(labels, logits, binary=False, aggregateBatch=True, aggregateEnsemble=True):
  """Average cross entropy for ensemble members (Gibbs cross entropy).

  For each datapoint (x,y), the ensemble's Gibbs cross entropy is:

  ```
  - (1/ensemble_size) sum_{m=1}^ensemble_size log p(y|x,theta_m).
  ```

  The Gibbs cross entropy approximates the average cross entropy of a single
  model drawn from the (Gibbs) ensemble.

  Args:
    labels: tf.Tensor of shape [...].
    logits: tf.Tensor of shape [ensemble_size, ..., num_classes].
    binary: bool, whether it is a binary classification (sigmoid as activation).
    aggregateBatch: bool, whether or not to average over the batch.
    aggregateEnsemble: bool, whether or not to average over the ensemble.

  Returns:
    tf.Tensor of shape [...].
  """
  logits = tf.convert_to_tensor(logits)
  if binary:
    nll = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)),
        logits=logits
    )
  else:
    labels = tf.cast(labels, tf.int32)
    nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)[:-1]),
        logits=logits)

  if aggregateBatch:
    nll = tf.reduce_mean(nll, axis=list(range(1, len(nll.shape))))

  if aggregateEnsemble:
    nll = tf.reduce_mean(nll, axis=0)

  return nll


def aggregateMetrics(metrics):

    aggregated = {}

    for measure in metrics[list(metrics)[0]].keys():
        aggregated['corruption/'+measure] = np.mean([metrics[metric][measure].result() for metric in metrics.keys()])

    return aggregated
