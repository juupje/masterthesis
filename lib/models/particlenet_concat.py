import tensorflow as tf
from tensorflow import keras
from time import time
import numpy as np
from utils import activations
from .ScalingLayer import ScalingLayer
from .particlenet import dist_mat, features_of_knn, create_edge_conv, create_decoder
  
def create_particle_net_concat(coords, features, mask, conv_params : tuple, fc_params : tuple, name : str = "ParticleNet"):
    with tf.name_scope(name):
        mask = tf.cast(tf.not_equal(mask, 0),dtype='float32')
        shift = tf.multiply(999., tf.cast(tf.equal(mask,0), dtype='float32'))
        features = [tf.squeeze(keras.layers.BatchNormalization(name=f"{name}/Features_BN")(tf.expand_dims(features,axis=2)), axis=2)]
        #edge conv layers!
        coords = tf.add(shift,coords)
        features.append(create_edge_conv(coords, features[-1], k=conv_params[0]['k'], channels=conv_params[0]['channels'], activation=conv_params[0].get("activation", "relu"),name=f"{name}/EdgeConv1"))
        for idx,conv_param in enumerate(conv_params[1:]):
            coords = tf.add(shift, features[-1])
            features.append(create_edge_conv(coords, features[-1], k=conv_param['k'], channels=conv_param['channels'], scaling=conv_param.get("scaling", None), activation=conv_param.get("activation", "relu"), name=f"{name}/EdgeConv{idx+2}"))
        
        features = tf.concat(features, axis=-1) #(N,P,C0+C1+C2+...)
        print("Feature shape:", features.shape)
        features = tf.multiply(features, mask) #(N,P,C)
        #average pooling
        x = tf.reduce_mean(features, axis=1) #(N,C)
        
        #MLP layers
        if(fc_params is not None):
            x = create_decoder(x, fc_params, name=f"{name}/Decoder") #(N,2)
        return x
    
def particle_net_concat(input_shapes, convolutions, fcs, model_class=keras.Model) -> keras.Model:
    coords = keras.Input(name='coordinates', shape=input_shapes["coordinates"])
    features = keras.Input(name='features', shape=input_shapes["features"])
    mask = keras.Input(name='mask', shape=input_shapes["mask"])
    outputs = create_particle_net_concat(coords,features,mask, convolutions, fcs)
    return model_class(inputs=[coords,features,mask], outputs=outputs, name='ParticleNet')