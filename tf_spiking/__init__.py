from .integrate_and_fire import DenseLIF, DenseLIFCategory, LIF_Activation, DenseLIFNoSpike, GetEnd
from .preprocessing import IntensityToPoissonSpiking, IntensityToSpikeLatency, IntensityTile

from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    "DenseLIF": DenseLIF,
    "DenseLIFCategory": DenseLIFCategory,
    "IntensityToPoissonSpiking": IntensityToPoissonSpiking,
    "IntensityToSpikeLatency": IntensityToSpikeLatency,
    "IntensityTile": IntensityTile,
    "LIF_Activation": LIF_Activation,
    "DenseLIFNoSpike": DenseLIFNoSpike,
    "GetEnd": GetEnd,
})
