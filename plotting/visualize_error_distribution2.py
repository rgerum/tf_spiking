import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.keras as keras
from tensorflow.keras import layers, models, utils, callbacks
import tensorflow as tf
import numpy as np
from tf_spiking import DenseLIF, DenseLIFCategory, IntensityToPoissonSpiking
import matplotlib.pyplot as plt
import pylustrator
#pylustrator.start()

for plot_index in range(3):

    num_classes = 10
    model = models.Sequential([
        layers.Input((211, 211)),
        DenseLIF(1, surrogate="flat"),
    ])
    print(model.summary())
    if plot_index == 1:
        model.layers[0].layers[0].set_weights([model.weights[0].numpy() *0 + 0.3, model.weights[1].numpy()])
        model.layers[0].layers[1].set_weights([np.ones(())*1, model.weights[1].numpy()*0.1])
        model.layers[0].layers[0].set_weights([model.weights[0].numpy() *0 + 0.15, model.weights[1].numpy()])
        model.layers[0].layers[1].set_weights([np.ones(())*1, model.weights[1].numpy()*0.02])

    if plot_index == 0:
        model.layers[0].layers[0].set_weights([model.weights[0].numpy() *0 + 0.099, model.weights[1].numpy()])
        model.layers[0].layers[1].set_weights([np.ones(()), model.weights[1].numpy()*0.0])

    if plot_index == 2:
        model.layers[0].layers[0].set_weights([model.weights[0].numpy() *0 + 0.5, model.weights[1].numpy()])
        model.layers[0].layers[1].set_weights([np.ones(()), model.weights[1].numpy()*0.2])

    model2 = models.Sequential([
        layers.Input((211, 211)),
        DenseLIF(1, surrogate="flat", return_potential=True),
    ])
    model2.set_weights(model.get_weights())

    input = np.zeros((211, 211))
    t = 0
    j = 0
    spike_times = []
    if 1:
        for i in range(1, 220):
            t += 1
            j += np.abs(np.cos(t/100*np.pi*2*4.3)*10)+1
            print(int(j))
            try:
                input[int(j), int(j)] = 1
                spike_times.append(int(j))
            except IndexError:
                break
    else:
        for i in range(1, 220):
            j += 10
            print(int(j))
            try:
                input[int(j), int(j)] = 1
                spike_times.append(int(j))
            except IndexError:
                break

    input = np.zeros((211, 211))
    t = 0
    j = 0
    spike_times = []
    for i in range(1, 220):
        t -= 1
        if t <= 0:
            t = 11
        j += t
        print(int(j))
        try:
            input[int(j), int(j)] = 1
            spike_times.append(int(j))
        except IndexError:
            break
    #plt.plot(input[:, 0])


    y = model2.predict(input[None])

    yt = np.zeros(211)
    try:
        yt[np.where(y[0, :, 0]> 1)[0][1]] = 1
    except IndexError:
        try:
            yt[np.where(y[0, :, 0] > 1)[0][0]] = 1
        except IndexError:
            pass

    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        logits = model.layers[0](input[None])

        # Loss value for this batch.
        #loss = keras.losses.binary_crossentropy(yt[None, :, None], logits)
        loss = yt[None, :, None] * logits

    # Get gradients of weights wrt the loss.
    gradients = tape.gradient(loss, model.layers[0].layers[0].trainable_weights)
    np.sum(gradients[0])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0
    def plot(spikes, width=0., color=None):
        global color_index
        indices = np.where(spikes >= 0.1)[0]
        c = colors[color_index] if color is None else colors[color % len(colors)]
        for i in indices:
            print(i, spikes[i], width)
            print([i-width, i+width], [0, 0], [spikes[i], spikes[i]], c)
            plt.fill_between([i-width, i+width], [0, 0], [spikes[i], spikes[i]], edgecolor=c, facecolor=c, zorder=2)
        if color is not None:
            color_index = (color_index + 1) % len(colors)
    r = 5
    c = 3
    ax0 = plt.subplot(r, c, 1+c*0+plot_index)
    ax1 = plt.subplot(r, c, 1+c*1+plot_index)#,sharex=ax0)
    ax2 = plt.subplot(r, c, 1+c*2+plot_index)#,sharex=ax0, sharey=ax0)
    ax3 = plt.subplot(r, c, 1+c*3+plot_index)#,sharex=ax0, sharey=ax0)
    ax4 = plt.subplot(r, c, 1+c*4+plot_index)#,sharex=ax0, sharey=ax0)
    #ax5 = plt.subplot(r, c, 1+c*5+plot_index)#,sharex=ax0, sharey=ax0)

    for id, i in enumerate(spike_times):
        plt.sca(ax0)
        plot(input[i], color=id)

        plt.sca(ax4)
        plot(input[i]*gradients[0][i], 0.2, color=id)

    plt.sca(ax1)
    plt.plot(y[0], label="Vm", drawstyle="steps", zorder=0)

    plt.sca(ax2)
    plot((y[0, :, 0] >= 1)*1, 0.2, 1)#, label="output")

    plt.sca(ax3)
    plot(yt, 0.2, 1)#, label="target output")

    plt.xlabel("time steps")
    plt.ylim(0, 1.1)
#    plt.legend()

pylustrator.helper_functions.axes_to_grid()
#pylustrator.helper_functions.add_letters()
#plt.tight_layout()

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_position([0.099648, 0.817631, 0.251968, 0.148727])
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.500000, 1.002165])
plt.figure(1).axes[0].texts[0].set_text("integrator")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("input\nspikes")
plt.figure(1).axes[1].set_position([0.099648, 0.639159, 0.251968, 0.148727])
plt.figure(1).axes[1].get_yaxis().get_label().set_text("membrane\npotential")
plt.figure(1).axes[2].set_position([0.099648, 0.460687, 0.251968, 0.148727])
plt.figure(1).axes[2].get_yaxis().get_label().set_text("output\nspikes")
plt.figure(1).axes[3].set_position([0.099648, 0.282214, 0.251968, 0.148727])
plt.figure(1).axes[3].get_yaxis().get_label().set_text("spike\nerror")
plt.figure(1).axes[4].set_position([0.099648, 0.103742, 0.251968, 0.148727])
plt.figure(1).axes[4].get_xaxis().get_label().set_text("time steps")
plt.figure(1).axes[4].get_yaxis().get_label().set_text("error")
plt.figure(1).axes[5].set_position([0.402009, 0.817631, 0.251968, 0.148727])
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_ha("center")
plt.figure(1).axes[5].texts[0].set_position([0.504119, 1.002165])
plt.figure(1).axes[5].texts[0].set_text("mixed")
plt.figure(1).axes[6].set_position([0.402009, 0.639159, 0.251968, 0.148727])
plt.figure(1).axes[7].set_position([0.402009, 0.460687, 0.251968, 0.148727])
plt.figure(1).axes[8].set_position([0.402009, 0.282214, 0.251968, 0.148727])
plt.figure(1).axes[9].set_position([0.402009, 0.103742, 0.251968, 0.148727])
plt.figure(1).axes[9].get_xaxis().get_label().set_text("time stepstime steps")
plt.figure(1).axes[10].set_position([0.704370, 0.817631, 0.251968, 0.148727])
plt.figure(1).axes[10].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[10].transAxes)  # id=plt.figure(1).axes[10].texts[0].new
plt.figure(1).axes[10].texts[0].set_ha("center")
plt.figure(1).axes[10].texts[0].set_position([0.500000, 1.002165])
plt.figure(1).axes[10].texts[0].set_text("coincidence detector")
plt.figure(1).axes[11].set_position([0.704370, 0.639159, 0.251968, 0.148727])
plt.figure(1).axes[12].set_position([0.704370, 0.460687, 0.251968, 0.148727])
plt.figure(1).axes[13].set_position([0.704370, 0.282214, 0.251968, 0.148727])
plt.figure(1).axes[14].set_position([0.704370, 0.103742, 0.251968, 0.148727])
plt.figure(1).axes[14].get_xaxis().get_label().set_text("time steps")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.savefig(__file__[:-3]+".pdf")
plt.show()
