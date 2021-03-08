import tensorflow.keras as keras
import tensorflow.keras as K
from tensorflow.keras import layers, models, utils, callbacks

from tf_spiking import DenseLIF, DenseLIFCategory, IntensityToPoissonSpiking

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10
model = models.Sequential([
    keras.Input(x_train.shape[1:]),
    layers.Reshape([28*28]),
    IntensityToPoissonSpiking(50, 1/255),
    DenseLIF(128, surrogate="flat"),
    DenseLIFCategory(10),
])
print(model.summary())

grads = K.gradient(model.output, model.input)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# fit the model
model.fit(x_train, utils.to_categorical(y_train), batch_size=128*2, validation_data=(x_test, utils.to_categorical(y_test)), epochs=100, callbacks=[callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max')])
