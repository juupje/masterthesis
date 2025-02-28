"""
Original from: https://arxiv.org/pdf/2201.08187.pdf
The authors provided the PyTorch source code here: https://github.com/sdogsq/LorentzNet-release

This is a TensorFlow implementation of the same network. It is more-or-less identical to the original.
Note that some options might not be implemented or augmented with more flexible options.
Whereas the original code takes its settings from commandline arguments, this implementation uses
    python dictionaries which contain settings and their values as key/value pairs.

Author: Joep Geuskens
"""

import keras
from keras import ops as ko
import tensorflow as tf
from utils import activations
from .feature_layers import AdjacencyMatrix

def min_norm2(x):
    x2 = ko.square(x)
    #return x2[...,0]*2-ko.sum(x2, axis=-1)
    return ko.take(x2, 0, axis=-1)*2-ko.sum(x2, axis=-1)
def min_prod(x, y):
    z = x*y
    #return z[...,0]*2 - ko.sum(z, axis=-1)
    return ko.take(z, 0, axis=-1)*2 - ko.sum(z, axis=-1)
def minkowski_feats(x, edges):
    i, j = edges
    x_i, x_j = ko.take(x, i,axis=0), ko.take(x, j,axis=0) #(K,4)
    #shapes: (K,1), (K,1)
    return ko.expand_dims(min_norm2(x_i-x_j),axis=-1), ko.expand_dims(min_prod(x_i, x_j),axis=-1)

def psi(x):
    return ko.sign(x)*ko.log(ko.abs(x)+1)

def create_edge_mask(node_mask):
    edge_mask = ko.logical_and(ko.expand_dims(node_mask,axis=1), ko.expand_dims(node_mask, axis=2))
    diag_mask = ~ko.expand_dims(ko.eye(edge_mask.shape[1], dtype="bool"),axis=0)
    edge_mask = ko.logical_and(edge_mask,diag_mask)
    return edge_mask

def segment_mean(x, segment_ids, num_segments, sorted=False):
    seg_sum = ko.segment_sum(x, segment_ids=segment_ids, num_segments=num_segments, sorted=sorted)
    seg_N = ko.segment_sum(ko.ones_like(x), segment_ids=segment_ids, num_segments=num_segments, sorted=sorted)
    return seg_sum/seg_N

def create_mlp(x, params : tuple, name : str="MLP"):
    for idx, layer in enumerate(params):
        x = keras.layers.Dense(layer["N"], use_bias=layer.get("bias", True), activation=None, name=f"{name:s}.Dense-{idx:d}")(x)
        x = activations.get_activation(layer.get("activation",None))(x)
        if(layer.get("batchnorm", False)):
            x = keras.layers.BatchNormalization(name=f"{name:s}.BN-{idx:d}")(x)
    return x

class MLP(keras.layers.Layer):
    def __init__(self, params:tuple, name:str="MLP", **kwargs):
        super().__init__(name=name, **kwargs)
        self.params = params
        self.layers = []
    
    def get_config(self):
        config = super().get_config()
        config.update({"params": self.params})
        return config
    
    def build(self, input_shape):
        for idx, layer in enumerate(self.params):
            self.layers.append(keras.layers.Dense(layer["N"], use_bias=layer.get("bias", True), activation=None, name=f"{self.name:s}.Dense-{idx:d}"))
            self.layers.append(activations.get_activation(layer.get("activation",None)))
            if(layer.get("batchnorm", False)):
                self.layers.append(keras.layers.BatchNormalization(name=f"{self.name:s}.BN-{idx:d}"))
    
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class LGEB(keras.layers.Layer):
    def __init__(self, c : float, mlp_params : dict, compute_x : bool=True, use_psi:bool=False,name : str="LGEB", **kwargs):
        super().__init__(name=name, **kwargs)
        """
        Parameters
        ----------
        x : tensor(N*P,4)
            The physical 4-momentum of the particles in the events
        h : tensor(N*P,C)
            The C features of the particles in the events
        edges: list[tensor(K,), tensor(K,)]
            An adjancency list of nodes in the batch. K is corresponds to the number
                of unmasked particles in the batch
        c : float
            Scaling hyperparameter used in the dot product attention of x
        mlp_params: dict:
            A dictionary providing parameters used creating the MLP's used in the LGEB
        name : str,
            name of the layer
        """
        self.c = c
        self.name = name
        self.mlp_params = mlp_params
        self.compute_x = compute_x
        self.use_psi = use_psi

    def get_config(self):
        config = super().get_config()
        config.update({"c": self.c, "mlp_params": self.mlp_params, "compute_x": self.compute_x, "use_psi": self.use_psi})
        return config
    
    def build(self, input_shape):
        self.phi_e = MLP(params=self.mlp_params["phi_e"], name=f"{self.name:s}.phi_e")
        self.phi_m = MLP(params=self.mlp_params["phi_m"], name=f"{self.name:s}.phi_m")
        self.phi_h = MLP(params=self.mlp_params["phi_h"], name=f"{self.name:s}.phi_h")
        if self.compute_x:
            self.phi_x = MLP(params=self.mlp_params["phi_x"], name=f"{self.name:s}.phi_x")
            self.output_shape = [(None,4), (None, self.mlp_params["phi_h"][-1]["N"])]
        else:    
            self.output_shape = (None, self.mlp_params["phi_h"][-1]["N"])

    def compute_output_shape(self):
        return self.output_shape

    def call(self, x, h, edges):
        P = x.shape[1] #number of particles
        #calculate the minkowski norms and products
        norms, prods = minkowski_feats(x, edges) #(K,1), (K,1)
        if(self.use_psi):
            norms, prods = psi(norms), psi(prods)
        i, j = edges
        if(h is not None):
            #NOTE: tf.gather(h, i) can be read as h[i]
            phi_e_input = tf.concat((tf.gather(h, i), tf.gather(h, j), norms, prods),axis=-1) #(K,2H+2)
        else:
            phi_e_input = tf.concat((norms, prods),axis=-1) #(K,2)

        #m_ij = phi_e(h_i, h_j, psi(|x_i-x_j|^2), psi(<x_i,x_j>))
        m = self.phi_e(phi_e_input) #(K,2H+2) -> (K,H_m)
        w = self.phi_m(m) #(K,1)
        wm = tf.math.unsorted_segment_sum(w*m, segment_ids=i, num_segments=tf.shape(x)[0]) #(N*P,H_m)

        if(h is not None):
            dh = self.phi_h(tf.concat((h, wm),axis=-1)) #(N*P,H+H_m) -> (N*P,H)
            h = tf.add(h,dh) #(N*P,H)
        else:
            h = self.phi_h(wm) #(N*P,H_m) -> (N*P,H)

        if(self.compute_x):
            phi_x = self.phi_x(m) #(K,H_m) -> (K,1)
            #We use unsorted_segment_mean with segments i to sum over all neighbors of i
            x = tf.add(x, self.c*tf.math.unsorted_segment_mean(tf.multiply(phi_x,tf.gather(x,j)), segment_ids=i, num_segments=tf.shape(x)[0])) #(N*P,4)
            return h,x
        return h

class LorentzNetDecoder(keras.layers.Layer):
    def __init__(self, params:dict, name:str="LorentzNetDecoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.params = params
        self.decoder = []

    def get_config(self):
        config = super().get_config()
        config.update({"params": self.params})
        return config
    
    def build(self, input_shape):
        for idx, fc_param in enumerate(self.params):
            if("dropout" in fc_param):
                self.decoder.append(keras.layers.Dropout(fc_param['dropout'],name=f"{self.name}.Dropout_{idx+1}"))
            self.decoder.append(keras.layers.Dense(fc_param['N'], activation=None, name=f"{self.name}.Dense_{idx+1}"))
            act = fc_param.get('activation',None)
            self.decoder.append(activations.get_activation(fc_param.get('activation',None),name=f"{self.name}.{act}_{idx+1}"))
            if(fc_param.get("batchnorm", False)):
                self.decoder.append(keras.layers.BatchNormalization(name=f"{self.name}.BN_{idx+1}"))
    
    def call(self, x, y=None):
        if y is not None:
            x = ko.concatenate([x, y], axis=1)
        for layer in self.decoder:
            x = layer(x)
        return x

class LorentzNet(keras.layers.Layer):
    def __init__(self, params:dict, mlp_params:dict, features:int=None, decoder_inputs:int=None, name="LorentzNetV2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.params = params
        self.mlp_params = mlp_params
        self.n_features = features
        self.n_decoder_inputs = decoder_inputs
        
    def build(self, mom4_shape, mask_shape=None, *args):
        input_shape = (mom4_shape, mask_shape) + args
        #for some reason, keras only passes the first shape
        # luckily, we don't need the mask shape here!
        print("LorentzNet build shapes:", input_shape)
        if len(input_shape) >= 2:
            mom4, mask = input_shape[:2]
        else:
            raise ValueError("LorentzNet requires at least two inputs: mom4 and mask")
        if self.n_features:
            features = mom4[:-1] + (self.n_features,)
        else:
            features = None

        #embedding
        if(features is not None):
            emb = self.params["embedding"]
            if(isinstance(emb, dict)):
                dim, init, act = emb["dim"], emb.get("initializer", "glorot_uniform"), emb.get("activation", None)
                if(init == "hadamard"):
                    from utils import hadamard
                    init = hadamard.hadamard_weights
            else:
                dim, init, act = emb, "glorot_uniform", None
            self.embedding = keras.layers.Dense(dim, name=f"{self.name:s}.Scalar_Embedding", kernel_initializer=init, activation=act) #(N,P,H)

        #LGEBs
        self.LGEBs = []
        for i in range(self.params["L"]-1):
            self.LGEBs.append(LGEB(c=self.params['c'], mlp_params=self.mlp_params, use_psi=self.params.get("use_psi", False), name=f"{self.name}.LGEB{i+1}"))
        self.LGEBs.append(LGEB(c=self.params['c'], mlp_params=self.mlp_params, use_psi=self.params.get("use_psi", False), compute_x=False, name=f"{self.name}.LGEB{self.params['L']}"))

        #decoder
        self.decoder = None
        if("decoder" in self.params):
            self.decoder = LorentzNetDecoder(params=self.params["decoder"], name=f"{self.name}.Decoder")

    def call(self, mom4, mask, scalars=None, decoder_input=None):
        #prepare mask
        mask = ko.squeeze(ko.not_equal(mask, 0),axis=2) #0 if mask==0 else 1, shape: (N,P)
        edges = AdjacencyMatrix(ko.shape(mom4)[1])(mask, create_edge_mask(mask))
        mask = ko.cast(mask,dtype="float32")
        print("Edges:", edges)
        #embedding
        if scalars is not None:
            scalars = self.embedding(scalars) #(N,P,H)

        #LGEB layers!
        h = ko.reshape(scalars, (-1, ko.shape(scalars)[2])) if scalars is not None else None #(N*P,C)
        x = ko.reshape(mom4,  (-1, ko.shape(mom4)[2])) #(N*P,4)
        print("LGEB input shapes:", (x.shape, h.shape if h is not None else None))
        for i in range(len(self.LGEBs)-1):
            h,x = self.LGEBs[i](x, h, edges)
        h = self.LGEBs[-1](x, h, edges)

        mask = ko.reshape(mask, (-1,1))
        features = ko.multiply(h, mask) #(N*P,C)
        features = ko.reshape(features, (-1, ko.shape(mom4)[1], ko.shape(features)[1])) #unfold the batch and particle dimensions
        #average pooling
        x = ko.mean(features, axis=1) #(N,C)
        
        #Decoding layers
        if("decoder" in self.params):
            x = self.decoder(x, decoder_input)
        return x

def create_decoder(x:keras.KerasTensor, params:list|tuple, softmax:bool=True, name:str="Decoder") -> keras.KerasTensor:
    for idx, fc_param in enumerate(params):
        if("dropout" in fc_param):
            x = keras.layers.Dropout(fc_param['dropout'],name=f"{name}.Dropout_{idx+1}")(x)
        x = keras.layers.Dense(fc_param['N'], activation=None, name=f"{name}.Dense_{idx+1}")(x)
        act = fc_param.get('activation',None)
        x = activations.get_activation(fc_param.get('activation',None),name=f"{name}.{act}_{idx+1}")(x)
        if(fc_param.get("batchnorm", False)):
            x = keras.layers.BatchNormalization(name=f"{name}.BN_{idx+1}")(x)
    #Softmax
    if(softmax): #also add softmax
        x = keras.layers.Dense(2, activation='softmax',name=f"{name}.Dense_{len(fc_param)+1}")(x) #(N,2)
    return x

def create_lorentz_net(coords, scalars, mask, decoder_input, params : dict, mlp_params : dict, name : str = "LorentzNet"):
    # coords: (N,P,4)
    # scalars: (N,P,C)
    # mask: (N,P,1)
    # do some tests
    assert mlp_params["phi_x"][-1]["N"]==1, "phi_x must end with a layer with N=1"
    assert mlp_params["phi_m"][-1]["N"]==1, "phi_m must end with a layer with N=1"
    emb = params["embedding"]["dim"] if type(params["embedding"]) is dict else params["embedding"]
    assert mlp_params["phi_h"][-1]["N"]==emb, \
        "phi_h must end with a layer equal to scalar embedding dimension"
    
    with keras.name_scope(name):
        #prepare mask
        mask = ko.squeeze(ko.not_equal(mask, 0),axis=2) #0 if mask==0 else 1, shape: (N,P)
        edges = AdjacencyMatrix(ko.shape(coords)[1])(mask, create_edge_mask(mask))
        mask = ko.cast(mask,dtype="float32")
        #embedding
        if(scalars is not None):
            emb = params["embedding"]
            if(isinstance(emb, dict)):
                dim, init, act = emb["dim"], emb.get("initializer", "glorot_uniform"), emb.get("activation", None)
                if(init == "hadamard"):
                    from utils import hadamard
                    init = hadamard.hadamard_weights
            else:
                dim, init, act = emb, "glorot_uniform", None
            scalars = keras.layers.Dense(dim, name=f"{name:s}.Scalar_Embedding", kernel_initializer=init, activation=act)(scalars) #(N,P,H)

        #LGEB layers!
        h = ko.reshape(scalars, (-1, ko.shape(scalars)[2])) if scalars is not None else None #(N*P,C)
        x = ko.reshape(coords,  (-1, ko.shape(coords)[2])) #(N*P,4)
        print("LGEB input shapes:", (x.shape, h.shape if h is not None else None))
        #for i in range(params["L"]-1):
        #    h,x = create_LGEB(x, h, edges, segments, c=params['c'], mlp_params=mlp_params, use_psi=params.get("use_psi", False), name=f"{name}.LGEB{i+1}")
        #h = create_LGEB(x, h, edges, segments, c=params['c'], mlp_params=mlp_params, use_psi=params.get("use_psi", False), compute_x=False, name=f"{name}.LGEB{params['L']}")
        
        for i in range(params["L"]-1):
            h,x = LGEB(c=params['c'], mlp_params=mlp_params, use_psi=params.get("use_psi", False), name=f"{name}.LGEB{i+1}")(x, h, edges)
        h = LGEB(c=params['c'], mlp_params=mlp_params, use_psi=params.get("use_psi", False), compute_x=False, name=f"{name}.LGEB{params['L']}")(x, h, edges)
        

        mask = ko.reshape(mask, (-1,1))
        features = ko.multiply(h, mask) #(N*P,C)
        features = ko.reshape(features, (-1, ko.shape(coords)[1], ko.shape(features)[1])) #unfold the batch and particle dimensions
        #average pooling
        x = ko.mean(features, axis=1) #(N,C)
        
        #Decoding layer
        if("decoder" in params):
            if decoder_input is not None:
                x = ko.concatenate([x, decoder_input], axis=1)
            x = create_decoder(x,params["decoder"], name=f"{name}.Decoder")
        return x
    
def lorentz_net(input_shapes : dict, ln_params : dict, mlp_params : dict, no_scalars : bool=False, model_class=keras.Model, **kwargs):
    coords = keras.Input(name='coordinates', shape=input_shapes["coordinates"])
    mask = keras.Input(name='mask', shape=input_shapes["mask"])
    if no_scalars:
        scalars = None
        inputs = [coords,mask]
    else:
        scalars = keras.Input(name='scalars', shape=input_shapes["scalars"])
        inputs = [coords, scalars, mask]
    if(decoder_input_shape := input_shapes.get("decoder_input", None)):
        decoder_input = keras.Input(name="decoder_input", shape=decoder_input_shape)
        inputs.append(decoder_input)
    else:
        decoder_input = None
    
    outputs = LorentzNet(params=ln_params, mlp_params=mlp_params, name='LorentzNet')(coords, mask, scalars=scalars, decoder_input=decoder_input)
    #outputs = LorentzNetDecoder(params=ln_params["decoder"], name='Decoder')(ln)
    #outputs = create_lorentz_net(coords,scalars,mask, decoder_input=decoder_input, params=ln_params, mlp_params=mlp_params)
    return model_class(inputs=inputs, outputs=outputs, name='LorentzNet', **kwargs)