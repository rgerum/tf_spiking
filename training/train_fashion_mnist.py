import tensorflow.keras as keras
from tensorflow.keras import layers, models, utils, callbacks

from tf_spiking import DenseLIF, DenseLIFCategory, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory

# load the dataset
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# generate the model
num_classes = 10
model = models.Sequential([
    keras.Input(x_train.shape[1:]),
    layers.Reshape([28*28]),
    IntensityToPoissonSpiking(100, 1/255),
    DenseLIF(256, dt=0.1, surrogate="flat"),
    DenseLIFCategory(10),
])
print(model.summary())

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# fit the model
output_path = "fashion_mnist256_100_1_255_128_2_dt01_different_leak"
model.fit(x_train, utils.to_categorical(y_train), batch_size=128*2, validation_data=(x_test, utils.to_categorical(y_test)), epochs=100,
          callbacks=[callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max'),
                     callbacks.ModelCheckpoint(output_path+"/weights.hdf5", save_best_only=True, save_weights_only=True),
                     TrainingHistory(output_path)
                     ])
