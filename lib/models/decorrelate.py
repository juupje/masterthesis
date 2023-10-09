import tensorflow as tf
from tensorflow import keras
from time import time
import numpy as np
from utils import activations

#from https://blog.paperspace.com/absolute-guide-to-tensorflow/
@tf.custom_gradient
def grad_reverse(x, scaling:float=1.0):
    y = tf.identity(x)
    
    def custom_grad(dy):
        return -dy*scaling
    
    return y, custom_grad

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, scaling:float=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scaling = scaling

    def get_config(self):
        config = super().get_config()
        config.update({"scaling": self.scaling})
        return config
        
    def call(self, x):
        return grad_reverse(x, self.scaling)

def create_adversary(input_shape, params:dict, name:str="Adversary"):
    input_layer = tf.keras.layers.Input(shape=input_shape, name=f"{name}/Input")
    layer1 = params["layers"][0]
    x = tf.keras.layers.Dense(layer1.get("nodes", 10), activation=layer1.get("activation", None), name=f"{name}/Dense{i}")(x)
    for i, layer in enumerate(params["layers"][1:]):
        x = tf.keras.layers.Dense(layer.get("nodes", 10), activation=layer.get("activation", None), name=f"{name}/Dense{i+1}")(x)
    x = tf.keras.layers.Dense(params["output"].get("nodes", 10), activation=params["output"].get("activation", "softmax"), name=f"{name}/output")(x)
    return tf.keras.Model(inputs=input_layer,outputs=x, name=name)

def create_model(classifier:tf.keras.Model, grad_scaling:float, lr_ratio:float, adversary_params:dict={}, name:str="Decorrelated Model"):
    #get output of classifier
    input_layers = filter(lambda l: type(l)==tf.keras.Input, classifier.layers)

    new_inputs = [tf.keras.layers.Input(shape=l.shape[1:], name=l.name) for l in input_layers]
    classifier_output = classifier(new_inputs)
    
    gradrev = GradientReversalLayer(grad_scaling*lr_ratio, name="GradientReversal")(classifier_output)
    adversary = create_adversary(gradrev.shape, adversary_params)
    adversary_output = adversary(gradrev)

    model = tf.keras.Model(inputs=new_inputs, outputs=[classifier_output, adversary_output], name=name)
    return model