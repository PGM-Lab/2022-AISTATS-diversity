import numpy as np
import tensorflow as tf
from baselines.cifar import varianceBound

K = 10
N = 1000


class VarianceBoundTest(tf.test.TestCase):
    def testBound1(self):
        probs = tf.random.uniform(shape=[K, N])

        v1 = varianceBound.logVarianceBoundPAC2B1ForLoop(tf.math.log(probs))
        v2 = varianceBound.logVarianceBoundPAC2B1(tf.math.log(probs),probs.shape[0])

        self.assertAlmostEqual(v1[0].numpy(), v2[0].numpy())
        self.assertAlmostEqual(v1[1].numpy(), v2[1].numpy())

    def testBound2(self):
        probs = tf.random.uniform(shape=[K, N])
        log_likelihood = tf.math.log(probs)

        mean = tf.reduce_mean(probs, axis=0)
        max = tf.reduce_max(probs, axis=0)
        inc = tf.math.log(mean) - tf.math.log(max)
        inc = tf.clip_by_value(inc, clip_value_min=-10., clip_value_max=-0.01)
        h1 = tf.math.divide(inc, tf.square(1 - tf.math.exp(inc))) + 1. / tf.multiply(tf.math.exp(inc),
                                                                                       1 - tf.math.exp(inc))

        logmax = tf.stop_gradient(tf.reduce_max(log_likelihood, axis=0))
        logmean = tf.reduce_logsumexp(log_likelihood, axis=0) - tf.math.log(tf.keras.backend.cast_to_floatx(probs.shape[0]))
        inc = logmean - logmax
        inc = tf.clip_by_value(inc, clip_value_min=-10., clip_value_max=-0.01)
        constant2 = tf.keras.backend.constant(2.)
        h2 = tf.stop_gradient(
            inc / tf.math.pow(1 - tf.math.exp(inc), constant2) + tf.math.pow(tf.math.exp(inc) * (1 - tf.math.exp(inc)),
                                                                             -1))
        self.assertAllClose(h1, h2, atol=1e-5)


    def testBound3(self):
        probs = tf.random.uniform(shape=[K, N])

        v1 = varianceBound.varianceBoundTight(tf.math.log(probs))
        v2 = varianceBound.logVarianceBoundPAC2B1(tf.math.log(probs),K)

        self.assertAlmostEqual(v1[0].numpy(), v2[0].numpy())
        self.assertAlmostEqual(v1[1].numpy(), v2[1].numpy())

    def testBound4(self):
        probs = tf.random.uniform(shape=[K, N])

        v1 = varianceBound.varianceBoundTight(tf.math.log(probs))
        v2 = varianceBound.logVarianceBoundPAC2B1(tf.math.log(probs),K)

        self.assertAlmostEqual(v1[0].numpy(), v2[0].numpy())
        self.assertAlmostEqual(v1[1].numpy(), v2[1].numpy())

    def testBound5(self):
        probs = tf.keras.backend.constant(np.array([0.999, 0.999]))

        v1 = varianceBound.varianceBoundTight(tf.math.log(probs))
        v2 = varianceBound.logVarianceBoundPAC2B1(tf.math.log(probs),2)

        self.assertAlmostEqual(v1[0].numpy(), tf.reduce_mean(tf.math.log(probs)))
        self.assertAlmostEqual(v2[0].numpy(), tf.reduce_mean(tf.math.log(probs)))

        self.assertAlmostEqual(v1[1].numpy(), 0)
        self.assertAlmostEqual(v2[1].numpy(), 0)

if __name__ == '__main__':
  tf.test.main()
