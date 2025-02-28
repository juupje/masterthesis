import keras
from keras import ops as ko, KerasTensor
from utils import activations
from .ScalingLayer import ScalingLayer
from .feature_layers import KNN_Features

def dist_mat(X : KerasTensor) -> KerasTensor:
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
    with keras.name_scope("distmat"):
        X2 = ko.sum(X*X, axis=2, keepdims=True) #(B,P,1)
        XXT = ko.matmul(X,ko.transpose(X,axes=(0,2,1)))
        return X2-2*XXT+ko.transpose(X2,axes=(0,2,1))

def features_of_knn(features:KerasTensor, knn_indices:KerasTensor, k:int) -> KerasTensor:
    """
    Collects the features of the k-nearest-neighbors of each particle of each event.
    
    Parameters
    ----------
    features : tensor(N,P,C)
        features of this batch
    knn_indices : tensor(N,P,K)
        the K indices of the K-nearest-neighbors of the particles in the events
    k: int
        K
    Returns
    -------
    tensor(N,P,K,C)
        The C features of the K nearest neighbors of the particles in the events
    """
    with keras.name_scope("knn_fts"):
        #TODO: somehow get N = features.shape[0]
        N = ko.sum(ko.ones_like(features[:,0,0], dtype='int16'))
        print(N)
        P = features.shape[1]
        #we need to do some crazy stuff to get the right tensor containing the indices of the feature
        # vectors which we need for each particle
        #create a tensor of indices indicating the event to which each of the knn's belongs
        b_idx = ko.tile(ko.reshape(ko.arange(N, dtype='int16'), (N,1,1,1)), (1,P,k,1)) #(N,P,k,1)
        #now, create a tensor which, for each particle in each event, k pairs of indices [event_number, particle_number]
        #where particle_number corresponds to it's k-th nearest neighbor's index in the event
        idx = ko.concatenate([b_idx, ko.expand_dims(knn_indices, axis=3)], axis=3) #(N,P,k,2)
        return ko.take_along_axis(features, idx) #now collect the features of those k nearest neighbors


def create_edge_conv(coords:KerasTensor, features:KerasTensor,
                     k : int, channels : tuple, scaling:str=None, activation:str='relu',name : str="EdgeConv") -> KerasTensor:
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
    with keras.name_scope(name):
        dists = dist_mat(coords)
        knn_fts = KNN_Features(k, name=f"{name}.KNN_Features")(features, dists)
        #for each edge connecting a particle to its knn's copy the particle's features
        knn_fts_center = ko.tile(ko.expand_dims(features, axis=2), (1,1,k,1)) #(N,P,K,C)
        #construct the features of the edges
        # each edge has two parts: x_i, the feature vector of the node and x_j-x_i, the distance to its knn
        x = ko.concatenate([knn_fts_center, ko.subtract(knn_fts, knn_fts_center)],axis=-1) #(N,P,K,2C)
        #x = edge features
        for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1,1),strides=1,data_format="channels_last", use_bias=False,
                                    kernel_initializer='glorot_normal', name=f"{name}.Conv2D{idx}")(x)
            x = keras.layers.BatchNormalization(name=f"{name}.BN{idx}")(x)
            x = activations.get_activation(activation,name=f"{name}.{activation}{idx}")(x)
        
        #average pooling over the neighbors
        x = ko.mean(x, axis=2) #(N,P,C[-1])
        
        #apply the scaling
        if(scaling):
            x = ScalingLayer(scaling, name=f"{name}.Scaling")(x)

        #add the shortcut
        shortcut = keras.layers.Conv2D(channels[-1], kernel_size=(1,1), strides=1,data_format="channels_last", use_bias=False,
                                    kernel_initializer="glorot_normal", name=f"{name}.Shortcut_Conv")(ko.expand_dims(features,axis=2))
        shortcut = keras.layers.BatchNormalization(name=f"{name}.Shortcut_BN")(shortcut)
        shortcut = ko.squeeze(shortcut,axis=2) #(N,P,C[-1])
        x = activations.get_activation(activation, name=f"{name}.{activation}_Last")(x+shortcut)
        return x

def create_decoder(x:KerasTensor, fc_params:tuple, softmax=True, name="Decoder") -> KerasTensor:
    for idx,fc_param in enumerate(fc_params):
        x = keras.layers.Dense(fc_param['nodes'], activation=None, name=f"{name}.Dense{idx+1}")(x)
        if("activation" in fc_params):
            x = activations.get_activations(fc_params["activation"],name=f"{name}.{fc_params['activation']}_{idx+1}")(x)
        if("dropout" in fc_param):
            x = keras.layers.Dropout(fc_param['dropout'],name=f"{name}.Dropout_{idx+1}")(x)
    if(softmax):
        return keras.layers.Dense(2, activation='softmax', name=f"{name}.Dense_{idx+1}")(x) #(N,2)
    return x
    
def create_particle_net(coords:KerasTensor, features:KerasTensor, mask:KerasTensor, decoder_input:KerasTensor,
                        conv_params : tuple, fc_params : tuple, name : str = "ParticleNet") -> KerasTensor:
    with keras.name_scope(name):
        mask = ko.cast(ko.not_equal(mask, 0),dtype='float32')
        shift = ko.multiply(999., ko.cast(ko.equal(mask,0), dtype='float32'))
        features = ko.squeeze(keras.layers.BatchNormalization(name=f"{name}.Features_BN")(ko.expand_dims(features,axis=2)), axis=2)
        #edge conv layers!
        coords = ko.add(shift,coords)
        features = create_edge_conv(coords, features, k=conv_params[0]['k'], channels=conv_params[0]['channels'], activation=conv_params[0].get("activation", "relu"),name=f"{name}.EdgeConv1")
        for idx,conv_param in enumerate(conv_params[1:]):
            coords = ko.add(shift, features)
            features = create_edge_conv(coords, features, k=conv_param['k'], channels=conv_param['channels'], scaling=conv_param.get("scaling", None), activation=conv_param.get("activation", "relu"), name=f"{name}.EdgeConv{idx+2}")
        
        features = ko.multiply(features, mask) #(N,P,C)
        #average pooling
        x = ko.mean(features, axis=1) #(N,C)
        
        #MLP layers
        if(fc_params is not None):
            if decoder_input is not None:
                x = ko.concatenate([x, decoder_input], axis=1)
            x = create_decoder(x, fc_params, name=f"{name}.Decoder") #(N,2)
        return x
    
def particle_net(input_shapes, convolutions, fcs, model_class=keras.Model, **kwargs) -> keras.Model:
    coords = keras.Input(name='coordinates', shape=input_shapes["coordinates"])
    features = keras.Input(name='features', shape=input_shapes["features"])
    mask = keras.Input(name='mask', shape=input_shapes["mask"])
    inputs=[coords,features,mask]
    if "decoder_input" in input_shapes:
        decoder_input = keras.Input(name='decoder_input', shape=input_shapes["decoder_input"])
        inputs.append(decoder_input)
    else:
        decoder_input = None
    outputs = create_particle_net(coords,features,mask, decoder_input=decoder_input, conv_params=convolutions, fc_params=fcs)
    return model_class(inputs=inputs, outputs=outputs, name='ParticleNet', **kwargs)