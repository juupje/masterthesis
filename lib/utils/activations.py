import keras
import re
from typing import Callable
def get_activation(activation:str, name:str=None) -> keras.layers.Layer|Callable:
    if activation is None:
        return keras.activations.linear
    if activation.startswith("leakyrelu"):
        match = re.match(r"leakyrelu\(([+-]?(\d*\.)?\d+)\)", activation)
        if(match):
            return keras.layers.LeakyReLU(alpha=float(match.group(1)),name=name)
        else:
            return keras.layers.LeakyReLU(name=name)
    return keras.activations.get(activation)