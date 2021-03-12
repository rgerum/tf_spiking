import tensorflow.keras as keras
from tensorflow.keras import layers, models, utils, callbacks

from tf_spiking import DenseLIF, DenseLIFNoSpike, IntensityToPoissonSpiking
from tf_spiking.helper import TrainingHistory


# load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# generate the model
num_classes = 10
i = 0.01
dt = 0.005
time_steps = 100
model = models.Sequential([
        keras.Input(x_train.shape[1:]),
        layers.Reshape([28*28]),
        IntensityToPoissonSpiking(time_steps, 1/255, dt=dt),
        DenseLIF(128, surrogate="flat", dt=dt),
        DenseLIFNoSpike(10),
    ])
print(model.summary())

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# fit the model
output_path = f"mnist_time{time_steps}_dt{dt}_poisson_hidden{128}"
model.fit(x_train, utils.to_categorical(y_train), batch_size=256, validation_data=(x_test, utils.to_categorical(y_test)), epochs=30,
          callbacks=[callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max'),
                     callbacks.ModelCheckpoint(output_path+"/weights.hdf5", save_best_only=True, save_weights_only=True),
                     TrainingHistory(output_path)
                     ])
