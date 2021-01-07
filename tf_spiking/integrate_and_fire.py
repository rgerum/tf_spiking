import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, TimeDistributed, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers

import tensorflow as tf
from .surrogates import get, theta_forget


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

    def __init__(self, w_input=1, w_leak=0.1, t_thresh=1, surrogate="flat", beta=10, return_potential=False, **kwargs):
        super().__init__(**kwargs)
        self.w_input_start = np.asarray(w_input)
        self.w_leak_start = np.asarray(w_leak)
        self.thresh = t_thresh

        self.theta = get(surrogate, beta)
        self.beta = beta
        self.surrogate = surrogate
        self.return_potential = return_potential

    def build(self, input_shape):
        self.w_input = self.add_weight(
            name='w_input',
            shape=self.w_input_start.shape,
            initializer=tf.keras.initializers.Constant(self.w_input_start),
            trainable=True)

        self.w_leak = self.add_weight(
            name='w_leak',
            shape=self.w_leak_start.shape,
            initializer=tf.keras.initializers.RandomUniform(0, self.w_leak_start),
            trainable=True)

        #self.set_weights([self.w_input_start, self.w_leak_start])
        super().build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        return {"w_input": self.w_input_start, "w_leak": self.w_leak_start, "t_thresh": self.thresh, "beta": self.beta, "surrogate": self.surrogate}

    def call(self, x):
        time_steps = x.shape[1]
        #channels = self.w_input.shape[0]
        pre_channels = x.shape[2]

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


class DenseLIF(Sequential):
    def __init__(self, units, dt=1, surrogate="flat", beta=10, return_potential=False):
        super().__init__([
            TimeDistributed(Dense(units)),
            LIF_Activation(w_input=dt, w_leak=np.ones(units)*0.1*dt, surrogate=surrogate, beta=beta, return_potential=return_potential),
        ])

class DenseLIFCategory(Sequential):
    def __init__(self, units, surrogate="flat", beta=10):
        super().__init__([
            TimeDistributed(Dense(units)),
            LIF_Activation(w_leak=0.1, surrogate=surrogate, beta=beta),
            SumEnd(),
        ])
