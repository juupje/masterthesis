import numpy as np
from abc import abstractmethod

_features = {}
def decoderfeature(func):
    _features[func.__name__.lower()] = func
    return func

class DecoderFeature:
    """
    Calculates a feature based on the input data so that it can be used in the decoder part of a network.
    """
    def __init__(self, dimension, datasets=None):
        self.dimension = dimension
        self.required_datasets = datasets or []
        if type(self.required_datasets) is str:
            self.required_datasets = [self.required_datasets]

    @abstractmethod
    def __call__(self, data:dict, extra_data:dict={}):
        raise NotImplementedError("Subclasses must implement this method")

@decoderfeature
class NParticles(DecoderFeature):
    def __init__(self):
        super().__init__(2)

    def __call__(self, data:dict, extra_data:dict={}):
        return np.stack((np.sum(data["jet1"]["mask"], axis=(1,2)),np.sum(data["jet2"]["mask"], axis=(1,2))),axis=1)

class Mjet(DecoderFeature):
    def __init__(self, jets):
        super().__init__(len(jets) if isinstance(jets, (list,tuple)) else 1, ["jet_coords"])
        self.jets = jets

    def __call__(self, data:dict, extra_data:dict = {}):
        if 'jet_coords' in data:
            d = data["jet_coords"]
        else:
            d = extra_data["jet_coords"]
        return d[:,self.jets,[3]]

@decoderfeature
class Mjet1(Mjet):
    def __init__(self):
        super().__init__(0)

@decoderfeature
class Mjet2(Mjet):
    def __init__(self):
        super().__init__(1)

@decoderfeature
class JetSeparation(DecoderFeature):
    def __init__(self):
        super().__init__(2, ["jet_coords"])

    def __call__(self, data:dict, extra_data:dict={}):
        if 'jet_coords' in data:
            d = data["jet_coords"]
        else:
            d = extra_data["jet_coords"]
        return np.stack((d[:, 0, 1]-d[:, 1, 1], d[:, 0, 2]-d[:, 1, 2]), axis=1)

def get(name) -> DecoderFeature|None:
    if name.lower() in _features:
        return _features[name.lower()]()