import keras
from keras import ops as ko
from keras import KerasTensor
from .particlenet import create_edge_conv, create_decoder
  
def create_particle_net_concat(coords:KerasTensor, features:KerasTensor, mask:KerasTensor, decoder_input:KerasTensor,
                               conv_params : tuple, fc_params : tuple, name : str = "ParticleNet") -> KerasTensor:
    """
    Similar to the normal particle net, but concatenates the features of the edge convolutions before feeding them into the decoder
    """
    with keras.name_scope(name):
        mask = ko.cast(ko.not_equal(mask, 0),dtype='float32')
        shift = ko.multiply(999., ko.cast(ko.equal(mask,0), dtype='float32'))
        features = [ko.squeeze(keras.layers.BatchNormalization(name=f"{name}.Features_BN")(ko.expand_dims(features,axis=2)), axis=2)]
        #edge conv layers!
        coords = ko.add(shift,coords)
        features.append(create_edge_conv(coords, features[-1], k=conv_params[0]['k'], channels=conv_params[0]['channels'], activation=conv_params[0].get("activation", "relu"),name=f"{name}.EdgeConv1"))
        for idx,conv_param in enumerate(conv_params[1:]):
            coords = ko.add(shift, features[-1])
            features.append(create_edge_conv(coords, features[-1], k=conv_param['k'], channels=conv_param['channels'], scaling=conv_param.get("scaling", None), activation=conv_param.get("activation", "relu"), name=f"{name}.EdgeConv{idx+2}"))
        
        features = ko.concatenate(features, axis=-1) #(N,P,C0+C1+C2+...)
        print("Feature shape:", features.shape)
        features = ko.multiply(features, mask) #(N,P,C)
        #average pooling
        x = ko.mean(features, axis=1) #(N,C)
        
        #MLP layers
        if(fc_params is not None):
            if decoder_input is not None:
                x = ko.concatenate([x, decoder_input], axis=1) #(N,C+D)
            x = create_decoder(x, fc_params, name=f"{name}.Decoder") #(N,2)
        return x
    
def particle_net_concat(input_shapes, convolutions, fcs, model_class=keras.Model, **kwargs) -> keras.Model:
    coords = keras.Input(name='coordinates', shape=input_shapes["coordinates"])
    features = keras.Input(name='features', shape=input_shapes["features"])
    mask = keras.Input(name='mask', shape=input_shapes["mask"])
    inputs=[coords,features,mask]
    if "decoder_input" in input_shapes:
        decoder_input = keras.Input(name='decoder_input', shape=input_shapes["decoder_input"])
        inputs.append(decoder_input)
    else:
        decoder_input = None
    outputs = create_particle_net_concat(coords,features,mask, decoder_input=decoder_input, conv_params=convolutions, fc_params=fcs)
    return model_class(inputs=inputs, outputs=outputs, name='ParticleNet', **kwargs)