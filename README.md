# Tensorflow Spiking Layer

This package provides spiking layers for the use in keras.

Spiking layers are implemented as leaky-integrate-and-fire (LIF) neurons, which are trained using a surrogate gradient.
For more details see [https://arxiv.org/abs/2004.13532][here].

[here]: https://arxiv.org/abs/2004.13532

## Example

This example illustrates how the package can be used to train a spiking network to classify the 
MNIST dataset.

```python
import tensorflow.keras as keras
from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking

# load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# generate the model
model = keras.models.Sequential([
        keras.Input(x_train.shape[1:]),
        keras.layers.Reshape([28*28]),
        IntensityToPoissonSpiking(100, 1/255, dt=0.005),
        DenseLIF(128, surrogate="flat", dt=0.005),
        DenseLIFNoSpike(10),
    ])
print(model.summary())

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# fit the model
model.fit(x_train, keras.utils.to_categorical(y_train), validation_data=(x_test, keras.utils.to_categorical(y_test)), 
          batch_size=256, epochs=30)
```