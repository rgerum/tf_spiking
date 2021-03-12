import tensorflow.keras as keras

from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory


# load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# generate the model
num_lif_units = 128
num_classes = 10
i = 0.01
dt = 0.005
time_steps = 100
model = keras.models.Sequential([
        keras.Input(x_train.shape[1:]),
        IntensityToPoissonSpiking(time_steps, 1/255, dt=dt),
        DenseLIF(num_lif_units, surrogate="flat", dt=dt),
        DenseLIFNoSpike(num_classes),
    ])
print(model.summary())

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# fit the model
output_path = f"mnist_time{time_steps}_dt{dt}_poisson_hidden{num_lif_units}"
model.fit(x_train, keras.utils.to_categorical(y_train), validation_data=(x_test, keras.utils.to_categorical(y_test)),
          batch_size=256, epochs=30,
          callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max'),
                     keras.callbacks.ModelCheckpoint(output_path+"/weights.hdf5", save_best_only=True, save_weights_only=True),
                     TrainingHistory(output_path)
                     ])
