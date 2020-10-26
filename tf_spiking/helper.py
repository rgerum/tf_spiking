from tensorflow.keras.layers import Dropout, MaxPool1D, MaxPool2D, Conv1D, Conv2D, Dense
from auditory_pathway import LIFLayer, BandpassLayer, HearingLossLayer, NoiseLayer, Cortex, MultiplyFactorLayer
from tensorflow.keras.models import load_model
from pathlib import Path


def writeShape(shape):
    return ", ".join(str(s) for s in shape[1:])

def getName(layer):
    if isinstance(layer, Conv2D):
        return "Convolution 2D"
    if isinstance(layer, Conv1D):
        return "Convolution 1D"
    if isinstance(layer, MaxPool2D):
        return "MaxPooling 2D"
    if isinstance(layer, MaxPool1D):
        return "MaxPooling 1D"
    if isinstance(layer, Dropout):
        return "DropOut"
    if isinstance(layer, Dense):
        return "Dense"
    return layer.__class__.__name__

def printModelInfo(model):
    if isinstance(model, (str, Path)):
        model = load_model(str(model))
    i = 0
    for layer in model.layers:
        if layer.name == "reshape_cast":
            continue
        try:
            active = layer.activation.__name__
        except AttributeError:
            active = "~"
        extra = "~"
        if isinstance(layer, Dropout):
            extra = f"droupout: {layer.rate}"
        if isinstance(layer, (MaxPool1D, MaxPool2D)):
            extra = f"poolsize: {layer.pool_size}"
        i += 1
        print(i, getName(layer), writeShape(layer.input.shape) + "; " + writeShape(layer.output.shape), active, extra, sep=" & ", end="\\\\\n")