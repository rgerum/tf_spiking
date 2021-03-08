import tensorflow as tf

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

import tensorflow.keras as keras
from tensorflow.keras import layers, models, utils, callbacks
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, TimeDistributed, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers

import tensorflow as tf
from tf_spiking.surrogates import get, theta_forget
from tf_spiking.integrate_and_fire import lif_gradient, DenseLIFNoSpike
import tensorflow as tf
from tf_spiking import DenseLIF, DenseLIFCategory, IntensityToPoissonSpiking, IntensityTile
from tf_spiking.integrate_and_fire import LIF_Activation, lif_gradient_membrane_potential
from tf_spiking.helper import TrainingHistory

import matplotlib.pyplot as plt

# load the dataset
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


def plotPotential(a):
    cmap = plt.get_cmap("viridis")
    cmap.set_bad('red')

    a[a > 1] = np.nan

    plt.imshow(a, interpolation='none', cmap=cmap, vmin=0, vmax=1)

def plot(model, x):
    results = []
    i = 0
    while True:
        model2 = models.Sequential()
        model2.add(layers.InputLayer(model.input.shape[1:]))
        j = 0
        for layer in model.layers:  # go through until last layer
            if isinstance(layer, layers.Lambda):
                continue
            if j > i:
                break
            if isinstance(layer, Sequential):
                submodel = Sequential()
                submodel.add(layers.InputLayer(layer.input.shape[1:]))
                for layer2 in layer.layers:
                    if j > i:
                        break
                    submodel.add(layer2)
                    j += 1
                model2.add(submodel)
            else:
                model2.add(layer)
                j += 1
            #model2.build()
        if isinstance(model2.layers[-1], LIF_Activation):
            model2.layers[-1].return_potential = True
        results.append(model2.predict(x_train[:1]))
        if isinstance(model2.layers[-1], LIF_Activation):
            model2.layers[-1].return_potential = False
        if j <= i:
            break
        i += 1
    plt.clf()
    plt.subplot(1, 6, 1)
    plt.imshow(x[0], cmap="gray")
    plt.subplot(1, 6, 2)
    plt.imshow(results[1][0].T)
    plt.subplot(1, 6, 3)
    plt.imshow(results[2][0].T)
    plt.subplot(1, 6, 4)
    plotPotential(results[3][0].T)
    plt.subplot(1, 6, 5)
    plt.imshow(np.isnan(results[3][0].T), vmin=0, vmax=1)
    plt.subplot(2, 6, 6)
    plt.plot(results[4][0])
    plt.subplot(2, 6, 12)
    #plt.imshow(np.isnan(results[5][0].T), vmin=0, vmax=1)
    plt.plot(results[5][0], "-")
    for i in range(10):
        plt.text(100, results[5][0][-1, i], i)
    #plt.plot(results[5][0])
    plt.tight_layout()

import pylustrator
#pylustrator.start()

for i in [0.01]:#[10, 20, 10, 5, 1]:#[0.1, 0.03, 0.01]:
    # generate the model
    num_classes = 10
    dt = 1/200
    model = models.Sequential([
        keras.Input(x_train.shape[1:]),
        layers.Reshape([28*28]),
        IntensityTile(100, 1/255.*0.1),
        #IntensityToPoissonSpiking(100, 1/255, dt=dt),
        layers.TimeDistributed(layers.Dense(64, use_bias=False)),
        LIF_Activation(
            units=64,
            input_initializer=tf.keras.initializers.Constant(1),
            leak_initializer=tf.keras.initializers.Constant(i*dt),
            #leak_initializer=tf.keras.initializers.RandomUniform(0, i*dt),
        ),
        DenseLIFNoSpike(10),
#        DenseLIFCategory(10),
    ])
    if 1:
        model.load_weights(rf"E:\tf_spiking\training\fashion_mnist_grid_repeat_units4_dt10_leak{i}_constant_trainable_nobias_nospike\weights.hdf5")#fashion_mnist_grid_dt10_leak{i}_uniform_fixed\weights.hdf5")
        plot(model, x_train[:1])
        #% start: automatic generated code from pylustrator
        plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
        import matplotlib as mpl
        plt.figure(1).axes[0].set_xlim(-0.5, 27.5)
        plt.figure(1).axes[0].set_ylim(27.5, -0.5)
        plt.figure(1).axes[0].set_xticks([0.0, 25.0])
        plt.figure(1).axes[0].set_yticks([np.nan])
        plt.figure(1).axes[0].set_xticklabels(["", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
        plt.figure(1).axes[0].set_yticklabels([""], fontsize=10)
        plt.figure(1).axes[0].set_position([0.006904, 0.879810, 0.086501, 0.115399])
        plt.figure(1).axes[1].set_ylim(783.5, -0.5)
        plt.figure(1).axes[1].set_yticks([np.nan])
        plt.figure(1).axes[1].set_yticklabels([""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="right")
        plt.figure(1).axes[1].set_position([0.107652, 0.051015, 0.090275, 0.944194])
        plt.figure(1).axes[2].set_position([0.282816, 0.438287, 0.154062, 0.526155])
        plt.figure(1).axes[2].spines['right'].set_visible(False)
        plt.figure(1).axes[2].spines['top'].set_visible(False)
        plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
        plt.figure(1).axes[2].texts[0].set_ha("center")
        plt.figure(1).axes[2].texts[0].set_position([0.527000, 1.012587])
        plt.figure(1).axes[2].texts[0].set_text("input")
        plt.figure(1).axes[3].set_yticklabels([])
        plt.figure(1).axes[3].get_yaxis().get_label().set_text('')
        plt.figure(1).axes[3].set_position([0.466858, 0.438287, 0.154062, 0.526155])
        plt.figure(1).axes[3].spines['right'].set_visible(False)
        plt.figure(1).axes[3].spines['top'].set_visible(False)
        plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
        plt.figure(1).axes[3].texts[0].set_ha("center")
        plt.figure(1).axes[3].texts[0].set_position([0.500000, 1.012587])
        plt.figure(1).axes[3].texts[0].set_text("$V_t$")
        plt.figure(1).axes[4].set_yticklabels([])
        plt.figure(1).axes[4].get_yaxis().get_label().set_text('')
        plt.figure(1).axes[4].set_position([0.650901, 0.438287, 0.154062, 0.526155])
        plt.figure(1).axes[4].spines['right'].set_visible(False)
        plt.figure(1).axes[4].spines['top'].set_visible(False)
        plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
        plt.figure(1).axes[4].texts[0].set_ha("center")
        plt.figure(1).axes[4].texts[0].set_position([0.448144, 1.012587])
        plt.figure(1).axes[4].texts[0].set_text("spiking $y_t$")
        plt.figure(1).axes[5].set_position([0.282816, 0.064023, 0.231880, 0.261015])
        plt.figure(1).axes[5].spines['right'].set_visible(False)
        plt.figure(1).axes[5].spines['top'].set_visible(False)
        plt.figure(1).axes[6].set_position([0.573082, 0.064023, 0.231880, 0.261015])
        plt.figure(1).axes[6].spines['right'].set_visible(False)
        plt.figure(1).axes[6].spines['top'].set_visible(False)
        #% end: automatic generated code from pylustrator
        plt.show()
    print(model.summary())
    #exit()
    # compile the model
    #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    output_path = f"fashion_mnist_grid_repeat_units4_dt10_leak{i}_constant_trainable_nobias_nospike"
    model.fit(x_train, utils.to_categorical(y_train), batch_size=256, validation_data=(x_test, utils.to_categorical(y_test)), epochs=100,
              callbacks=[callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max'),
                         callbacks.ModelCheckpoint(output_path+"/weights.hdf5", save_best_only=True, save_weights_only=True),
                         TrainingHistory(output_path)
                         ])

