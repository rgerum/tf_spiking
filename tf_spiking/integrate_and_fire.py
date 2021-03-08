import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, TimeDistributed, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers

import tensorflow as tf
from tf_spiking.surrogates import get, theta_forget


# define the lif
@tf.function
def lif_gradient(x, w_i, w_l, theta_one, t_thresh=1):
    time_steps = x.shape[1]

    Vm = w_i * x[:, 0] * 0
    states = tf.TensorArray(tf.float32, size=time_steps)

    for i in tf.range(time_steps):
        Vm = tf.nn.relu(w_i * x[:, i] + (1 - w_l) * Vm * theta_forget(t_thresh - Vm))
        spike = theta_one(Vm - t_thresh)
        states = states.write(i, spike)

    return tf.transpose(states.stack(), (1, 0, 2))


# define the lif
@tf.function
def lif_sum_no_spike(x, w_i, w_l):
    time_steps = x.shape[1]

    Vm = w_i * x[:, 0] * 0
    states = tf.TensorArray(tf.float32, size=time_steps)

    for i in tf.range(time_steps):
        Vm = tf.nn.relu(w_i * x[:, i] + (1 - w_l) * Vm)
        states = states.write(i, Vm)

    return tf.transpose(states.stack(), (1, 0, 2))


# define the lif
@tf.function
def lif_gradient_membrane_potential(x, w_i, w_l, theta_one, t_thresh=1):
    time_steps = x.shape[1]

    Vm = w_i * x[:, 0] * 0
    states = tf.TensorArray(tf.float32, size=time_steps)

    for i in tf.range(time_steps):
        Vm = tf.nn.relu(w_i * x[:, i] + (1 - w_l) * Vm * theta_forget(t_thresh - Vm))
        spike = theta_one(Vm - t_thresh)
        states = states.write(i, Vm)

    return tf.transpose(states.stack(), (1, 0, 2))



class LIF_Activation(Layer):

    def __init__(self, units=1, t_thresh=1, surrogate="flat", beta=10, return_potential=False,
                 input_initializer='random_uniform',
                 leak_initializer='random_uniform',
                 leak_constant=False,
                 no_spike=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.units_leak = units if leak_constant is False else 1
        self.thresh = t_thresh
        self.no_spike = no_spike

        self.theta = get(surrogate, beta)
        self.beta = beta
        self.surrogate = surrogate
        self.return_potential = return_potential

        if leak_initializer == "random_uniform":
            self.leak_initializer = tf.keras.initializers.RandomUniform(0, self.w_leak_start)
        elif leak_initializer == "constant":
            self.leak_initializer = tf.keras.initializers.Constant(self.w_leak_start)
        else:
            self.leak_initializer = initializers.get(leak_initializer)

        self.input_initializer = initializers.get(input_initializer)

    def build(self, input_shape):
        self.w_input = self.add_weight(
            name='w_input',
            shape=[self.units],
            initializer=self.input_initializer,
            trainable=False)

        self.w_leak = self.add_weight(
            name='w_leak',
            shape=[self.units_leak],
            initializer=self.leak_initializer,
            trainable=True)

        super().build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        return {
            "units": self.units,
            "leak_initializer": tf.keras.initializers.serialize(self.leak_initializer),
            "input_initializer": tf.keras.initializers.serialize(self.input_initializer),
            "t_thresh": self.thresh,
            "surrogate": self.surrogate,
            "beta": self.beta,
            "no_spike": self.no_spike,
        }

    def call(self, x):
        time_steps = x.shape[1]
        #channels = self.w_input.shape[0]
        pre_channels = x.shape[2]

        if self.no_spike:
            y = lif_sum_no_spike(x, self.w_input, self.w_leak)
            return y

        # integrate (batch, time_steps, pre_channels)
        if self.return_potential:
            y = lif_gradient_membrane_potential(x, self.w_input, self.w_leak, theta_one=self.theta, t_thresh=self.thresh)
        else:
            y = lif_gradient(x, self.w_input, self.w_leak, theta_one=self.theta, t_thresh=self.thresh)
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


class GetEnd(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x[:, -1, :]
        x = tf.expand_dims(tf.math.reduce_sum(x, axis=1), axis=1)
        x = x / (tf.expand_dims(tf.math.reduce_sum(x, axis=2), axis=2) + 0.001)
        return x[:, 0, :]



class DenseLIF(Sequential):
    def __init__(self, units, dt=1, surrogate="flat", beta=10, return_potential=False):
        super().__init__([
            TimeDistributed(Dense(units)),
            LIF_Activation(
                units=units,
                input_initializer=tf.keras.initializers.Constant(1*dt),
                leak_initializer=tf.keras.initializers.RandomUniform(0, 0.1*dt),
                surrogate=surrogate, beta=beta, return_potential=return_potential),
        ])

class DenseLIFCategory(Sequential):
    def __init__(self, units, dt=1, surrogate="flat", beta=10):
        super().__init__([
            TimeDistributed(Dense(units)),
            LIF_Activation(
                input_initializer=tf.keras.initializers.Constant(1 * dt),
                leak_initializer=tf.keras.initializers.Constant(0.1 * dt),
                surrogate=surrogate, beta=beta),
            SumEnd(),
        ])

class DenseLIFNoSpike(Sequential):
    def __init__(self, units, dt=1, surrogate="flat", beta=10):
        super().__init__([
            TimeDistributed(Dense(units)),
            LIF_Activation(
                input_initializer=tf.keras.initializers.Constant(1 * dt),
                leak_initializer=tf.keras.initializers.Constant(0 * dt),
                no_spike=True,
            ),
            GetEnd(),
            tf.keras.layers.Softmax(),
        ])