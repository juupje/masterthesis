"""
This is the most simple permutation invariant network I could come up with.
It is based on simple message passing and pooling operations.
"""

import keras
from keras import ops as ko, KerasTensor
from utils import activations
from models.masked_batch_normalization import MaskedBatchNormalization

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
    
def create_permu_net(coords:KerasTensor, features:KerasTensor, mask:KerasTensor, decoder_input:KerasTensor, params : dict,
                        decoder_params : tuple, name : str = "PermuNet") -> KerasTensor:
    with keras.name_scope(name):
        mask = ko.cast(ko.not_equal(mask, 0),dtype='float32')
        features = ko.squeeze(MaskedBatchNormalization(name=f"{name}.Features_BN")(ko.expand_dims(features, axis=2), ko.expand_dims(mask,axis=2)),axis=2) #(B,P,F)
        features = keras.layers.Dense(params["embedding"]['channels'], activation=params["embedding"]['activation'], name=f"{name}.Feature_Embedding")(features) #(B,P,C_E)
        
        for idx, layer in enumerate(params["layers"]):
            features_i = ko.expand_dims(features, axis=2) #(B,P,1,C_{idx-1})
            features_j = ko.expand_dims(features, axis=1) #(B,1,P,C_{idx-1})
            #compute features messages
            diff = ko.subtract(features_i, features_j) #(B,P,P,C_{idx-1})
            msg = keras.layers.Dense(layer["channels"], activation=layer["activation"], name=f"{name}.Msg{idx}")(diff) #(B,P,P,C_idx)
            #apply mask and average over the particles
            msg = ko.multiply(msg, ko.expand_dims(mask, axis=-1))
            msg = ko.sum(msg, axis=2)/ko.sum(mask,axis=(1,2), keepdims=True) #(B,P,C_idx)
            #update the features
            combined = ko.concatenate([features, msg], axis=-1) #(B,P,2C_{idx-1}))
            features = keras.layers.Dense(layer["channels"], activation=layer["activation"], name=f"{name}.Update{idx}")(combined)

            #update the features using the coords
            coords_update = keras.layers.Dense(layer["channels"], activation=layer["activation"], name=f"{name}.Coords_Update{idx+1}")(coords) #(B,P,C_idx)
            coords_update = ko.multiply(coords_update, layer["update_scale"]) #(B,P,C_idx)
            coords_update = ko.multiply(coords_update, mask) #(B,P,C_idx)*(B,P,1)
            features = ko.add(features, coords_update) #(B,P,C_idx)+(B,P,C_idx)
            features = keras.layers.Dropout(rate=layer["dropout"])(features) #(B,P,C_idx)
            #update the coords using the features
            features_update = keras.layers.Dense(ko.shape(coords)[2], activation=layer["activation"], name=f"{name}.Feat_Update{idx+1}")(features) #(B,P,2)
            features_update = ko.multiply(features_update, layer["update_scale"]) #(B,P,2)
            features_update = ko.multiply(features_update, mask) #(B,P,2)*(B,P,1)
            coords = ko.add(coords, features_update) #(B,P,2)+(B,P,2)
            
        x = ko.multiply(features, mask) #(B,P,C_L)
        #average pooling
        x = ko.mean(x, axis=1) #(B,C_L)
        #MLP layers
        if(decoder_params is not None):
            if decoder_input is not None:
                x = ko.concatenate([x, decoder_input], axis=1) #(B,C_L+D)
            x = create_decoder(x, decoder_params, name=f"{name}.Decoder") #(N,2)
        return x
    
def permunet(input_shapes, params, decoder, model_class=keras.Model, **kwargs) -> keras.Model:
    coords = keras.Input(name='coordinates', shape=input_shapes["coordinates"])
    features = keras.Input(name='features', shape=input_shapes["features"])
    mask = keras.Input(name='mask', shape=input_shapes["mask"])
    inputs = [coords, features, mask]
    if "decoder_input" in input_shapes:
        decoder_input = keras.Input(name='decoder_input', shape=input_shapes["decoder_input"])
        inputs.append(decoder_input)
    else:
        decoder_input = None
    outputs = create_permu_net(coords,features,mask, decoder_input=decoder_input, params=params, decoder=decoder)
    return model_class(inputs=inputs, outputs=outputs, name='PermuNet', **kwargs)