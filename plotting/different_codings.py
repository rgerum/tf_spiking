import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import tensorflow.keras as keras
from tensorflow.keras import layers, models, utils, callbacks

from tf_spiking import DenseLIF, DenseLIFCategory, IntensityToPoissonSpiking, IntensityToSpikeLatency
from tf_spiking.helper import TrainingHistory

# load the dataset
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# generate the model
num_classes = 10
model = models.Sequential([
    keras.Input(x_train.shape[1:]),
    layers.Reshape([28*28]),
    IntensityToSpikeLatency(100, t_eff=4),
#    IntensityToPoissonSpiking(100, 1/255, dt=0.01),
])
print(model.summary())

y = model.predict(x_train[:1])
import numpy as np
import matplotlib.pyplot as plt
plt.subplot(121)
plt.imshow(x_train[0])
plt.subplot(122)
y = y[0]
a, b = np.where(y)
plt.plot(b, a, "o")
plt.ylim(0, y.shape[0])
plt.xlabel("timestep")
plt.ylabel("unit")
plt.tight_layout()
#plt.imshow(y)
#plt.savefig("encoding_poisson.png")
plt.savefig("encoding_latency.png")
plt.show()
