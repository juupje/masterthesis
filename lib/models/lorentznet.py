"""
Original from: https://arxiv.org/pdf/2201.08187.pdf
The authors provided the PyTorch source code here: https://github.com/sdogsq/LorentzNet-release

This is a TensorFlow implementation of the same network. It is more-or-less identical to the original.
Note that some options might not be implemented or augmented with more flexible options.
Whereas the original code takes its settings from commandline arguments, this implementation uses
    python dictionaries which contain settings and their values as key/value pairs.

Author: Joep Geuskens
"""

import tensorflow as tf
from utils import activations
from utils import hadamard

def min_norm2(x):
    x2 = tf.square(x)
    return x2[...,0]*2-tf.reduce_sum(x2, axis=-1)
def min_prod(x, y):
    z = x*y
    return z[...,0]*2 - tf.reduce_sum(z, axis=-1)
def psi(x):
    return tf.math.sign(x)*tf.math.log(tf.math.abs(x)+1)

def create_mlp(x, params : tuple, name : str="MLP"):
    for idx, layer in enumerate(params):
        x = tf.keras.layers.Dense(layer["N"], use_bias=layer.get("bias", True), activation=None, name=f"{name:s}/Dense-{idx:d}")(x)
        x = activations.get_activation(layer.get("activation",None))(x)
        if(layer.get("batchnorm", False)):
            x = tf.keras.layers.BatchNormalization(name=f"{name:s}/BN-{idx:d}")(x)
    return x

def create_LGEB(x, h, c : float, mlp_params : dict, compute_x : bool=True, use_psi:bool=False,name : str="LGEB"):
    """
    Parameters
    ----------
    x : tensor(N,P,4)
        The physical coordinates of the particles in the events
    h : tensor(N,P,C)
        The C features of the particles in the events
    c : float
        Scaling hyperparameter used in the dot product attention of x
    mlp_params: dict:
        A dictionary providing parameters used creating the MLP's used in the LGEB
    name : str,
        name of the layer
    """
    with tf.name_scope("LGEB"):
        P = x.shape[1] #number of particles
        #calculate the minkowski norms and products
        #needed for proper broadcasting:
        x1 = tf.expand_dims(x,axis=2) #(N,P,1,4)
        x2 = tf.expand_dims(x,axis=1) #(N,1,P,4)
        norms = tf.expand_dims(min_norm2(x1-x2), axis=-1) #(N,P,P,1)
        prods = tf.expand_dims(min_prod(x1,x2), axis=-1) #(N,P,P,1)
        if(use_psi):
            norms, prods = psi(norms), psi(prods)
        if(h is not None):
            h1 = tf.expand_dims(h,axis=2) #(N,P,1,H)
            h1 = tf.tile(h1, (1,1,P,1))   #(N,P,P,H)
            h2 = tf.tile(tf.expand_dims(h,axis=1), (1,P,1,1)) #(N,P,P,H)
            phi_e_input = tf.concat((h1, h2, norms, prods),axis=-1) #(N,P,P,2H+2)
        else:
            phi_e_input = tf.concat((norms, prods),axis=-1) #(N,P,P,2)

        #m_ij = phi_e(h_i, h_j, psi(|x_i-x_j|^2), psi(<x_i,x_j>))
        m = create_mlp(phi_e_input, params=mlp_params["phi_e"],
                name=f"{name:s}/phi_e") #(N,P,P,2H+2) -> (N,P,P,H_m)
        w = create_mlp(m, params=mlp_params["phi_m"], name=f"{name:s}/phi_m") #(N,P,P,1)
        wm = tf.reduce_sum(w*m, axis=2) #(N,P,H_m)

        if(h is not None):
            dh = create_mlp(tf.concat((h, wm),axis=-1), params=mlp_params["phi_h"], name=f"{name:s}/phi_h") #(N,P,H+H_m) -> (N,P,H)
            h = tf.add(h,dh) #(N,P,H)
        else:
            h = create_mlp(wm, params=mlp_params["phi_h"], name=f"{name:s}/phi_h") #(N,P,H_m) -> (N,P,H)

        if(compute_x):
            phi_x = create_mlp(m, params=mlp_params["phi_x"],
                    name=f"{name:s}/phi_x") #(N,P,P,1)
            x = tf.add(x, c*tf.reduce_mean(tf.multiply(phi_x,tf.expand_dims(x,axis=1)),axis=2)) #(N,P,4)
            return h,x
        return h

def create_decoder(x:tf.Tensor, params:list|tuple, softmax:bool=True, name:str="Decoder") -> tf.Tensor:
    for idx, fc_param in enumerate(params):
        if("dropout" in fc_param):
            x = tf.keras.layers.Dropout(fc_param['dropout'],name=f"{name}/Dropout_{idx+1}")(x)
        x = tf.keras.layers.Dense(fc_param['N'], activation=None, name=f"{name}/Dense_{idx+1}")(x)
        act = fc_param.get('activation',None)
        x = activations.get_activation(fc_param.get('activation',None),name=f"{name}/{act}_{idx+1}")(x)
        if(fc_param.get("batchnorm", False)):
            x = tf.keras.layers.BatchNormalization(name=f"{name}/BN_{idx+1}")(x)
    #Softmax
    if(softmax): #also add softmax
        x = tf.keras.layers.Dense(2, activation='softmax',name=f"{name}/Dense_{len(fc_param)+1}")(x) #(N,2)
    return x

def create_lorentz_net(coords, scalars, mask, params : dict, mlp_params : dict, name : str = "LorentzNet"):
    # coords: (N,P,4)
    # scalars: (N,P,C)
    # mask: (N,P,1)
    # do some tests
    assert mlp_params["phi_x"][-1]["N"]==1, "phi_x must end with a layer with N=1"
    assert mlp_params["phi_m"][-1]["N"]==1, "phi_m must end with a layer with N=1"
    emb = params["embedding"]["dim"] if type(params["embedding"]) is dict else params["embedding"]
    assert mlp_params["phi_h"][-1]["N"]==emb, \
        "phi_h must end with a layer equal to scalar embedding dimension"
    
    with tf.name_scope(name):
        #prepare mask
        mask = tf.cast(tf.not_equal(mask, 0),dtype='float32') #0 if mask==0 else 1
        #embedding
        if(scalars is not None):
            emb = params["embedding"]
            if(isinstance(emb, dict)):
                dim, init, act = emb["dim"], emb.get("initializer", "glorot_uniform"), emb.get("activation", None)
                if(init == "hadamard"):
                    init = hadamard.hadamard_weights
            else:
                dim, init, act = emb, "glorot_uniform", None
            scalars = tf.keras.layers.Dense(dim, name=f"{name:s}/Scalar_Embedding", kernel_initializer=init, activation=act)(scalars) #(N,P,H)

        #LGEB layers!
        h, x = scalars, coords
        for i in range(params["L"]-1):
            h,x = create_LGEB(x, h, c=params['c'], mlp_params=mlp_params, use_psi=params.get("use_psi", False), name=f"{name}/LGEB{i+1}")
        h = create_LGEB(x, h, c=params['c'], mlp_params=mlp_params, use_psi=params.get("use_psi", False), compute_x=False, name=f"{name}/LGEB{params['L']}")
        
        features = tf.multiply(h, mask) #(N,P,C)
        #average pooling
        x = tf.reduce_mean(features, axis=1) #(N,C)
        
        #Decoding layer
        if("decoder" in params):
            x = create_decoder(x,params["decoder"], name=f"{name}/Decoder")
        return x
    
def lorentz_net(input_shapes : dict, ln_params : dict, mlp_params : dict, no_scalars : bool=False, model_class=tf.keras.Model):
    coords = tf.keras.Input(name='coordinates', shape=input_shapes["coordinates"])
    mask = tf.keras.Input(name='mask', shape=input_shapes["mask"])
    if no_scalars:
        scalars = None
        inputs = [coords,mask]
    else:
        scalars = tf.keras.Input(name='scalars', shape=input_shapes["scalars"])
        inputs = [coords, scalars, mask]
    outputs = create_lorentz_net(coords,scalars,mask, params=ln_params, mlp_params=mlp_params)
    return model_class(inputs=inputs, outputs=outputs, name='LorentzNet')