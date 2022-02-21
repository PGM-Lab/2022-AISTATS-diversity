import tensorflow as tf
from edward2.tensorflow import generated_random_variables

class TrainableNormalDuplicate(tf.keras.layers.Layer):
  """Random normal op as an initializer with trainable mean and stddev."""

  def __init__(self,
               initializer,
               **kwargs):
    """Constructs the initializer."""
    super(TrainableNormalDuplicate, self).__init__(**kwargs)
    self.initializer = initializer

  def build(self, shape, dtype=None):

    self.built = True

  def __call__(self, shape, dtype=None):
    if not self.initializer.built:
      self.initializer.build(shape, dtype)
    mean = self.initializer.mean
    if self.initializer.mean_constraint:
      mean = self.initializer.mean_constraint(mean)
    stddev = self.initializer.stddev
    if self.initializer.stddev_constraint:
      stddev = self.initializer.stddev_constraint(stddev)
    return generated_random_variables.Independent(
        generated_random_variables.Normal(loc=mean, scale=stddev).distribution,
        reinterpreted_batch_ndims=len(shape))
