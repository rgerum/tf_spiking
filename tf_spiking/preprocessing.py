import tensorflow as tf
from tensorflow.keras.layers import Layer


class IntensityToSpikeLatency(Layer):
    def __init__(self, N, t_eff=50e-3, theta=0.2, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.t_eff = t_eff
        self.theta = theta

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        return {"N": self.N, "t_eff": self.t_eff, "theta": self.theta}

    def call(self, x):
        # get the number of units
        M = x.shape[1]
        # flatten the input
        x = tf.reshape(x, [-1])
        # find the indices where the input is above the threshold
        indices = tf.where(x > self.theta)
        # calculate the time (in units of timesteps)
        T = tf.cast(self.t_eff * tf.math.log(x / (x - self.theta)) * self.N, tf.int64)
        # add an indices (first index goes over all rows with finite time, the second index specifies which timestep)
        indices = tf.transpose(tf.stack([indices[..., 0], tf.gather(T, indices)[..., 0]]))
        # create a [None, M, N] shape
        new_shape = tf.shape(tf.tile(tf.ones_like(x)[:, None], (1, self.N)))
        # scatter ones according to the times
        res = tf.scatter_nd(indices, tf.ones_like(indices[:, 0]), tf.cast(new_shape, tf.int64))
        # separate batch and number of units again and return
        return tf.reshape(res, [-1, M, self.N])


class IntensityToPoissonSpiking(Layer):
    def __init__(self, N, factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.factor = factor

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        return {"N": self.N, "factor": self.factor}

    def call(self, x):
        x = tf.cast(x[:, None, :], tf.float32)*self.factor
        y = tf.tile(x, (1, self.N, 1))
        return tf.random.uniform(shape=tf.shape(y), dtype=x.dtype) < x
