import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, TimeDistributed, Dense
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf


# define the custom theta functions
@tf.custom_gradient
def theta_forget(x):
    def grad(dy):
        return dy * 0

    return tf.cast(x >= 0, tf.float32), grad


@tf.custom_gradient
def theta_one(x):
    def grad(dy):
        return dy

    return tf.cast(x >= 0, tf.float32), grad


# define the lif
@tf.function
def lif_gradient(x, w_i, w_l, t_refrectory=1, V_thresh=1):
    time_steps = x.shape[1]

    Vm = w_i * x[:, 0] * 0
    R = w_i * x[:, 0] * 0
    states = tf.TensorArray(tf.float32, size=time_steps)

    for i in tf.range(time_steps):
        R = theta_forget(Vm - V_thresh) * t_refrectory + theta_forget(R - 1) * (R - 1)
        Vm = (w_i * x[:, i] + (1 - w_l) * Vm) * theta_forget(-R)
        spike = theta_one(Vm - V_thresh)
        states = states.write(i, Vm)

    return tf.transpose(states.stack(), (1, 0, 2))


# define the layer
class LIFLayer(Layer):

    def __init__(self, w_input=0.3, w_leak=0.1, t_thresh=1, **kwargs):
        super().__init__(**kwargs)
        self.w_input_start = np.asarray(w_input)
        self.w_leak_start = np.asarray(w_leak)
        self.thresh = t_thresh

    def build(self, input_shape):
        self.w_input = self.add_weight(
            name='w_input',
            shape=self.w_input_start.shape,
            trainable=True)

        self.w_leak = self.add_weight(
            name='w_leak',
            shape=self.w_leak_start.shape,
            trainable=True)

        self.set_weights([self.w_input_start, self.w_leak_start])
        super().build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        return {"w_input": self.w_input_start, "w_leak": self.w_leak_start, "t_thresh": self.thresh}

    def call(self, x):
        # x is in shape (batch, group, time_steps, channels)
        group = x.shape[1]
        time_steps = x.shape[2]
        channels = self.w_input.shape[0]
        pre_channels = x.shape[3]
        # reshape to (batch x group, time_steps, pre_channels)
        x = tf.reshape(x, (-1, time_steps, pre_channels))
        # integrate (batch x group, time_steps, pre_channels)
        y = lif_gradient(x, self.w_input, self.w_leak, self.thresh)
        # reshape to (batch, group, time_steps, channels)
        y = tf.reshape(y, (-1, group, time_steps, channels))
        return y


# define the layer
class LIFLayer_direct(Layer):

    def __init__(self, w_input=0.3, w_leak=0.1, t_thresh=1, **kwargs):
        super().__init__(**kwargs)
        self.w_input_start = np.asarray(w_input)
        self.w_leak_start = np.asarray(w_leak)
        self.thresh = t_thresh

    def build(self, input_shape):
        self.w_input = self.add_weight(
            name='w_input',
            shape=self.w_input_start.shape,
            trainable=True)

        self.w_leak = self.add_weight(
            name='w_leak',
            shape=self.w_leak_start.shape,
            trainable=True)

        self.set_weights([self.w_input_start, self.w_leak_start])
        super().build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        return {"w_input": self.w_input_start, "w_leak": self.w_leak_start, "t_thresh": self.thresh}

    def call(self, x):
        time_steps = x.shape[1]
        #channels = self.w_input.shape[0]
        pre_channels = x.shape[2]

        # integrate (batch, time_steps, pre_channels)
        y = lif_gradient(x, self.w_input, self.w_leak, self.thresh)
        # reshape to (batch, time_steps, channels)
        y = tf.reshape(y, (-1, time_steps, pre_channels))
        return y


class SumEnd(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = x[:, -10:, :]
        x = tf.expand_dims(tf.math.reduce_sum(x, axis=1), axis=1)
        x = x / (tf.expand_dims(tf.math.reduce_sum(x, axis=2), axis=2) + 0.001)
        return x[:, 0, :]


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


class DenseLIF(Sequential):
    def __init__(self, units):
        super().__init__([
            TimeDistributed(Dense(units)),
            LIFLayer_direct(w_leak=0.1),
        ])

class DenseLIFCategory(Sequential):
    def __init__(self, units):
        super().__init__([
            TimeDistributed(Dense(units)),
            LIFLayer_direct(w_leak=0.1),
            SumEnd(),
        ])
