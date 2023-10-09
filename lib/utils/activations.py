import tensorflow as tf
import re
from typing import Callable
def get_activation(activation:str, name:str=None) -> tf.keras.layers.Layer|Callable:
    if activation is None:
        return tf.keras.activations.linear
    if activation.startswith("leakyrelu"):
        match = re.match(r"leakyrelu\(([+-]?(\d*\.)?\d+)\)", activation)
        if(match):
            return tf.keras.layers.LeakyReLU(alpha=float(match.group(1)),name=name)
        else:
            return tf.keras.layers.LeakyReLU(name=name)
    return tf.keras.activations.get(activation)