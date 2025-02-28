"""
Originally from: https://arxiv.org/pdf/2211.00454.pdf
The authors provided the PyTorch source code here: https://github.com/abogatskiy/PELICAN

This is a TensorFlow implementation ofthe same architecture.
There are some differences to the original code, mostly to reduce the code's complexity.
Not all features and parameters are implemented.
Note that the arguments related to the number of nodes per layer are different, since TF does not
    require you to specify a layer's input shape. This gives a little bit more freedom in choosing parametes.
Also, the original code accepts all parameters through commandline inputs, but this implementation
    uses a python dictionary with the configured parameters as an input to the network's constructor.

Author: Joep Geuskens
"""

import keras
from keras import ops as ko, KerasTensor
from utils import activations
from utils.configs import marshal
from models import aggregation_functions as agg_fct
from models.pelican_layers import Eq2to0, Eq2to2, PelicanEmbedding
from models.masked_batch_normalization import MaskedBatchNormalization

#======== DEFAULT PARAMETERS ========
default_embed = {"dim": 20, "batchnorm":True}
default_2to2 = {"layers": [50,50,50,50,50], "message_layers": [[30],[30],[30],[30],[30]],
                "activation_agg": None, "activation_lin": 'leakyrelu', "activation_msg": 'leakyrelu',
                "dropout": 0.25, "batchnorm": True, "aggregations": "M","ir_safe": False,
                "avg_nparticles": 49, "factorize": True}
default_msg  = {"layers": [30], "activation": 'leakyrelu', "ir_safe": False, "batchnorm":True}
default_2to0 = {"dim": 30, "activation_agg": 'leakyrelu', "activation_lin": None,
                "ir_safe": False, "aggregations": "M",
                "avg_nparticles": 49}
default_decoder = {"layers": [2], "activation": 'leakyrelu', "activation_last": 'softmax', "ir_safe":False} #layers should end with 2
default_params = {"embedding": default_embed, "2to2": default_2to2, "message": default_msg,
                "dropout_msg": 0.25, "2to0": default_2to0, "dropout_out": 0.1, "decoder": default_decoder}

def min_prod(x:KerasTensor, y:KerasTensor) -> KerasTensor:
    """
    Calculates the minkowski product of the 4-momenta in each tensor

    Parameters
    ----------
    `x`: KerasTensor(N,P,4)
    `y`: KerasTensor(N,P,4)

    Returns
    -------
    KerasTensor(N,P,P,1)
    """
    z = x*y
    return ko.expand_dims(ko.take(z, 0, axis=-1)*2 - ko.sum(z, axis=-1),axis=-1)

def create_embedding(x:KerasTensor, edge_mask:KerasTensor, params:dict,name:str="Embedding") -> KerasTensor:
    """
    Creates the embedding block of the network. Done using the `PelicanEmbedding` layer.
    
    Parameters
    ----------
    `x`: KerasTensor(N,P,P,1)
    `edge_mask`: KerasTensor(N,P,P,1)
    `params`: dict with keys:
        `dim`: number of dimensions in the output (feature size) (default: 20)
        `batchnorm`: if True, apply a batchnormalization at the end of the block (default: True)
    
    Returns
    -------
    KerasTensor(N,P,P,H), H=params['dim']
    """
    with keras.name_scope(name):
        x = PelicanEmbedding(params["dim"],name=name)(x) #(N,P,P,dim)
        if(params["batchnorm"]):
            x = MaskedBatchNormalization(name=f"{name:s}.MaskedBatchNorm")(x, edge_mask, training=None) #(N,P,P,dim)
        else:
            x = ko.where(edge_mask, x, 0, name=f"{name:s}.mask") 
        return x

def create_message_block(x:KerasTensor, edge_mask:KerasTensor, layers:list, activation:str='leakyrelu',\
            batchnorm:bool=True, ir_safe:bool=False, name:str="MessageBlock") -> KerasTensor:
    """
    Creates a message block. This includes a number of `Dense` layers with an activation followed
    by and optional batchnormalization. After the `Dense` layers, a mask is applied
    
    Parameters
    ----------
    `x`: KerasTensor(N,P,P,H)
    `edge_mask`: KerasTensor(N,P,P,1)
    `layers`: list of ints, each int being the number of nodes in a layer.
            So, the number of layers is `len(layers)`.
    `activation`: activation function to apply after each layer (default: 'leakyrelu').
    `batchnorm`: if True, a batchnormalization is applied at the end of the block (default: True).
    `ir_safe`: if True, no biases are used (default: False).
    `name`: name of the block (default: 'MessageBlock')

    Returns
    -------
    `output`: KerasTensor(N,P,P,H2)
    """
    
    for dim in layers:
        x = keras.layers.Dense(dim, activation=None, use_bias=not ir_safe, name=f"{name}.Dense")(x) #(N,P,P,H3)
        if(activation): x = activations.get_activation(activation)(x)
    if(batchnorm):
        x = MaskedBatchNormalization(name=f"{name}.MaskedBatchNorm")(x,edge_mask)
    else:
        #apply the mask again
        x = ko.where(edge_mask, x, 0,name=f"{name}.Mask")
    return x

def create_2to2_blocks(x:KerasTensor, edge_mask:KerasTensor, nparticles:KerasTensor, params:dict,\
        name="2to2Blocks") -> KerasTensor:
    """
    Creates a sequence of 2to2 blocks.
    
    Parameters
    ----------
    `x`: KerasTensor(N,P,P,H1)
    `edge_mask`: KerasTensor(N,P,P,1)
    `nparticles`: KerasTensor(N,1,1)
    `params`: dict with keys:
        - `layers`: list of ints, each int being the output dimension of the aggregation block.
        - `message_layers`: list of list of ints, each list of ints being the input of a message block.
        - `batchnorm`: if True, there will be a batch normalization after each message block
        - `dropout`: if True, a dropout layer is applied after each message block
        - `ir_safe`: if True, there will be no biases
        - `activation_msg`: the activation function of the message blocks
        - `activation_lin`, `activation_agg`, `avg_nparticles`, `aggregations`: passed to `Eq2to2`
    Returns
    -------
    `output`: KerasTensor(N,P,P,H2), H2=params['layers'][-1]
    """
    with keras.name_scope(name):
        layers = params["layers"]
        for idx in range(len(layers)):
            #dim is the size of the output
            x = create_message_block(x, edge_mask, layers=params["message_layers"][idx], activation=params["activation_msg"],
                            batchnorm=params["batchnorm"], ir_safe=params["ir_safe"],name=f"{name}.Message{idx}")
            if("dropout" in params):
                x = keras.layers.Dropout(params["dropout"],name=f"{name}.Dropout{idx}")(x)
                #TODO: Why do they transpose before and after dropout? Seems not to make a difference
            x = Eq2to2(units=layers[idx], activation_agg=params["activation_agg"], activation_lin=params["activation_lin"], 
                    ir_safe=params["ir_safe"], avg_nparticles=params["avg_nparticles"], aggregations=params["aggregations"],name=f"{name}.Agg{idx}")\
                    ([x, edge_mask, nparticles])
        return x

def create_decoder(x:KerasTensor, params:dict,name:str="Decoder") -> KerasTensor:
    """
    Creates the decoder block, which consists of a sequence of linear layers with activations between them.
    The last layer has a potentially different activation.
    
    Parameters
    ----------
    `x`: KerasTensor(N,H4)
    `params`: dict with possible keys
        - `layers`: a list of ints where each int indicates the number of nodes in a layer.
            So there will be `len(layers)` layers. The last layer is the number of outputs/classes. (default: [2])
        - `activation`: the activation function to apply in between layers (default: 'leakyrelu')
        - `activation_last`: activation of the last layer (default: 'softmax')
        - `ir_safe`: if True, do not use biases (default: False)
    
    Returns
    -------
    `pred`: KerasTensor(N,num_classes)
        Predictions of the network
    """
    N = len(params["layers"])
    for idx, units in enumerate(params["layers"]):
        x = keras.layers.Dense(units, use_bias=not params["ir_safe"], 
                activation=None, name=f"{name}.Dense{idx}")(x)
        if(idx < N-1 and params["activation"] is not None):
            x = activations.get_activation(params["activation"],name=f"{name}.{params['activation']}_{idx}")(x)
        elif(idx == N-1 and params["activation_last"] is not None):
             x = activations.get_activation(params["activation_last"],name=f"{name}.{params['activation']}_{idx}")(x)
    return x #(N,2)

def create_pelican(coords:KerasTensor, mask:KerasTensor, decoder_input:KerasTensor, params:dict,name:str="PELICAN") -> KerasTensor:
    #Create the parameter dictionaries
    param_imbed = marshal(params.get("embedding"), default_embed, "embedding",quiet=True)
    param_2to2 = marshal(params.get("2to2"), default_2to2, "2to2",quiet=True)
    param_msg = marshal(params.get("message"), default_msg, "message",quiet=True)
    param_2to0 = marshal(params.get("2to0"), default_2to0, "2to0",quiet=True)
    param_decoder = marshal(params.get("decoder"), default_decoder, "decoder",quiet=True) if params.get("decoder", None) is not None else None
    assert len(param_2to2["layers"])==len(param_2to2["message_layers"]), \
            "Number of 2to2 layers should be equal to number of message layers"
    with keras.name_scope(name):
        #Determine the masks and number of particles
        edge_mask = ko.expand_dims(mask,axis=1)*ko.expand_dims(mask,axis=2) #(N,P,P,1), 0 when either of the particles should be masked
        edge_mask = ko.not_equal(edge_mask,0) #True if the mask is non-zero
        nparticles = ko.sum(mask,axis=1,keepdims=True) #(N,1,1)

        #Calculate pairwise invariant masses (minkowski dot products) and create their embeddings
        dots = min_prod(ko.expand_dims(coords,axis=1),ko.expand_dims(coords,axis=2)) #(N,P,P,1)
        embedded = create_embedding(dots, edge_mask, param_imbed, name=f"{name}.Embedding") #(N,P,P,H1)
        
        #Pass them through the equivariant 2to2 blocks (multiple message + Eq2to2-aggregation layers)
        x = create_2to2_blocks(embedded, edge_mask, nparticles, param_2to2, name=f"{name}.2to2Blocks") #(N,P,P,H2)
        
        #Now, a 2to0 block
        #First, the message block and dropout
        x = create_message_block(x, edge_mask, **param_msg, name=f"{name}.MessageBlock") #(N,P,P,H3)
        if(params.get("dropout_msg",False)):
            x = keras.layers.Dropout(params["dropout_msg"], name=f"{name}.Dropout_MSG")(x)
        #And aggregate the result using a 2to0 layer (to get scalar values)
        x = Eq2to0(units=param_2to0["dim"], activation_agg=param_2to0["activation_agg"], activation_lin=param_2to0["activation_lin"], 
                    ir_safe=param_2to0["ir_safe"], avg_nparticles=param_2to0["avg_nparticles"], aggregations=param_2to0["aggregations"],name=f"{name}.Eq2to0")\
                    ([x, nparticles]) #(N,H4)
        
        #Finally, apply another dropout and then pass the results through the decoder
        if(params.get("dropout_out",False)):
            x = keras.layers.Dropout(params["dropout_out"], name=f"{name}.Dropout_out")(x) #(N,H4)
        if(param_decoder):
            if decoder_input is not None:
                x = ko.concatenate([x, decoder_input], axis=1) #(N,H4+D)
            x = create_decoder(x, param_decoder, name=f"{name}.Decoder") #(N,2)
        return x

def pelican(input_shapes : dict, params : dict, model_class=keras.Model, **kwargs) -> keras.Model:
    coords = keras.Input(name='coordinates', shape=input_shapes["coordinates"])
    mask = keras.Input(name='mask', shape=input_shapes["mask"])
    inputs = [coords,mask]
    if "decoder_input" in input_shapes:
        decoder_input = keras.Input(name='decoder_input', shape=input_shapes["decoder_input"])
        inputs.append(decoder_input)
    else:
        decoder_input = None
    outputs = create_pelican(coords,mask,decoder_input=decoder_input, params=params)
    return model_class(inputs=inputs, outputs=outputs, name='PELICAN', **kwargs)