from .integrate_and_fire import DenseLIF, DenseLIFCategory
from .preprocessing import IntensityToPoissonSpiking, IntensityToSpikeLatency

from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    "DenseLIF": DenseLIF,
    "DenseLIFCategory": DenseLIFCategory,
    "IntensityToPoissonSpiking": IntensityToPoissonSpiking,
    "IntensityToSpikeLatency": IntensityToSpikeLatency,
})
