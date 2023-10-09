import tensorflow as tf
from tensorflow import keras
from time import time
import numpy as np
from utils import activations
from .ScalingLayer import ScalingLayer

def dist_mat(X : tf.Tensor) -> tf.Tensor:
    """
    Calculates the pair-wise distance of the points in X.
    
    Parameters
    ----------
    X : tensor(B,P,C)
        B=Batch size/number of events, P=number of particles,
        and C=number of coordinates(eta+phi or number of features)

    Returns
    -------
    D : tensor(B,P,P)
        `D[b,i,j]=|X[b,i,:]-X[b,j,:]|^2`
    """
    with tf.name_scope("distmat"):
        X2 = tf.reduce_sum(X*X, axis=2, keepdims=True) #(B,P,1)
        XXT = tf.matmul(X,X, transpose_b=True)
        return X2-2*XXT+tf.transpose(X2,perm=(0,2,1))

def features_of_knn(features, knn_indices, k):
    """
    Collects the features of the k-nearest-neighbors of each particle of each event.
    
    Parameters
    ----------
    features : tensor(N,P,C)
        features of this batch
    knn_indices : tensor(N,P,K)
        the K indices of the K-nearest-neighbors of the particles in the events
    
    Returns
    -------
    tensor(N,P,K,C)
        The C features of the K nearest neighbors of the particles in the events
    """
    with tf.name_scope("knn_fts"):
        N = tf.shape(features)[0]
        P = features.shape[1]
        #we need to do some crazy stuff to get the right tensor containing the indices of the feature
        # vectors which we need for each particle
        #create a tensor of indices indicating the event to which each of the knn's belongs
        b_idx = tf.tile(tf.reshape(tf.range(N), (N,1,1,1)), (1,P,k,1)) #(N,P,k,1)
        #now, create a tensor which, for each particle in each event, k pairs of indices [event_number, particle_number]
        #where particle_number corresponds to it's k-th nearest neighbor's index in the event
        idx = tf.concat([b_idx, tf.expand_dims(knn_indices, axis=3)], axis=3) #(N,P,k,2)
        return tf.gather_nd(features, idx) #now collect the features of those k nearest neighbors


def create_edge_conv(coords, features, k : int, channels : tuple, scaling:str=None, activation:str='relu',name : str="EdgeConv"):
    """
    Parameters
    ----------
    coords : tensor(N,P,2)
        The physical coordinates of the particles in the events
    features : tensor(N,P,C)
        The C features of the particles in the events
    k : int,
        number of nearest neighbors (excluding the node itself)
    channels : tuple of int,
        tuple of the size of the convolution's output channels
    name : str,
        name of the layer
    """
    with tf.name_scope("EdgeConv"):
        dists = dist_mat(coords)
        _,top_k = tf.math.top_k(-dists,k=k+1, sorted=True) #+1 to account for the node itself
        top_k = top_k[:,:,1:] #remove the node itself
        #for each node/particle, collect the features of its knn's
        knn_fts = features_of_knn(features, top_k, k) #(N,P,K,C)
        #for each edge connecting a particle to its knn's copy the particle's features
        knn_fts_center = tf.tile(tf.expand_dims(features, axis=2), (1,1,k,1)) #(N,P,K,C)
        #construct the features of the edges
        # each edge has two parts: x_i, the feature vector of the node and x_j-x_i, the distance to its knn
        x = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)],axis=-1) #(N,P,K,2C)
        #x = edge features
        for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1,1),strides=1,data_format="channels_last", use_bias=False,
                                    kernel_initializer='glorot_normal', name=f"{name}/Conv2D{idx}")(x)
            x = keras.layers.BatchNormalization(name=f"{name}/BN{idx}")(x)
            x = activations.get_activation(activation,name=f"{name}/{activation}{idx}")(x)
        
        #average pooling over the neighbors
        x = tf.reduce_mean(x, axis=2) #(N,P,C[-1])
        
        #apply the scaling
        if(scaling):
            x = ScalingLayer(scaling, name=f"{name}/Scaling")(x)

        #add the shortcut
        shortcut = keras.layers.Conv2D(channels[-1], kernel_size=(1,1), strides=1,data_format="channels_last", use_bias=False,
                                    kernel_initializer="glorot_normal", name=f"{name}/Shortcut_Conv")(tf.expand_dims(features,axis=2))
        shortcut = keras.layers.BatchNormalization(name=f"{name}/Shortcut_BN")(shortcut)
        shortcut = tf.squeeze(shortcut,axis=2) #(N,P,C[-1])
        x = activations.get_activation(activation, name=f"{name}/{activation}_Last")(x+shortcut)
        return x

def create_decoder(x:tf.Tensor, fc_params:tuple, softmax=True, name="Decoder") -> tf.Tensor:
    for idx,fc_param in enumerate(fc_params):
        x = keras.layers.Dense(fc_param['nodes'], activation=None, name=f"{name}/Dense{idx+1}")(x)
        if("activation" in fc_params):
            x = activations.get_activations(fc_params["activation"],name=f"{name}/{fc_params['activation']}_{idx+1}")(x)
        if("dropout" in fc_param):
            x = keras.layers.Dropout(fc_param['dropout'],name=f"{name}/Dropout_{idx+1}")(x)
    if(softmax):
        return keras.layers.Dense(2, activation='softmax', name=f"{name}/Dense_{idx+1}")(x) #(N,2)
    return x
    
def create_particle_net(coords, features, mask, conv_params : tuple, fc_params : tuple, name : str = "ParticleNet"):
    with tf.name_scope(name):
        mask = tf.cast(tf.not_equal(mask, 0),dtype='float32')
        shift = tf.multiply(999., tf.cast(tf.equal(mask,0), dtype='float32'))
        features = tf.squeeze(keras.layers.BatchNormalization(name=f"{name}/Features_BN")(tf.expand_dims(features,axis=2)), axis=2)
        #edge conv layers!
        coords = tf.add(shift,coords)
        features = create_edge_conv(coords, features, k=conv_params[0]['k'], channels=conv_params[0]['channels'], activation=conv_params[0].get("activation", "relu"),name=f"{name}/EdgeConv1")
        for idx,conv_param in enumerate(conv_params[1:]):
            coords = tf.add(shift, features)
            features = create_edge_conv(coords, features, k=conv_param['k'], channels=conv_param['channels'], scaling=conv_param.get("scaling", None), activation=conv_param.get("activation", "relu"), name=f"{name}/EdgeConv{idx+2}")
        
        features = tf.multiply(features, mask) #(N,P,C)
        #average pooling
        x = tf.reduce_mean(features, axis=1) #(N,C)
        
        #MLP layers
        if(fc_params is not None):
            x = create_decoder(x, fc_params, name=f"{name}/Decoder") #(N,2)
        return x
    
def particle_net(input_shapes, convolutions, fcs, model_class=keras.Model):
    coords = keras.Input(name='coordinates', shape=input_shapes["coordinates"])
    features = keras.Input(name='features', shape=input_shapes["features"])
    mask = keras.Input(name='mask', shape=input_shapes["mask"])
    outputs = create_particle_net(coords,features,mask, convolutions, fcs)
    return model_class(inputs=[coords,features,mask], outputs=outputs, name='ParticleNet')