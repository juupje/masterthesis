import keras
from keras import ops as ko
from utils.configs import marshal

def _combine_jets(jets, combiner:str='concat'):
    if combiner == "concat":
        return ko.concatenate(jets,axis=1)
    elif combiner == "sum":
        return ko.sum(ko.stack(jets,axis=1),axis=1)
    elif combiner == 'mean':
        return ko.mean(ko.stack(jets,axis=1),axis=1)
    elif combiner == 'conv':
        return keras.layers.Conv1D(filters=1, kernel_size=1, strides=1)(ko.stack(jets,axis=1))
    else:
        raise ValueError(f"Invalid combiner {combiner}")

def pelican(input_shapes : dict, params : dict, model_class:keras.Model=keras.Model, **kwargs) -> keras.Model:
    from .pelican import create_pelican, create_decoder, default_decoder
    decoder_params = marshal(params.get("decoder"), default_decoder, "decoder",quiet=True)
    params = params.copy()
    del params["decoder"]
    inputs,nets = [], []
    for idx in range(input_shapes["njets"]):   
        coords = keras.Input(name=f"jet{idx+1:d}_coordinates", shape=input_shapes["coordinates"])
        mask = keras.Input(name=f"jet{idx+1:d}_mask", shape=input_shapes["mask"])
        inputs.extend([coords,mask])
        nets.append(create_pelican(coords,mask, None, params=params, name=f"PELICAN-Jet{idx+1:d}")) #(N,C)
    
    x = _combine_jets(nets, combiner=kwargs.pop("combiner", "concat"))

    if(decoder_input_shape := input_shapes.get("decoder_input", None)) is not None:
        decoder_input = keras.Input(name="decoder_input", shape=decoder_input_shape)
        inputs.append(decoder_input)
        x = ko.concatenate([x, decoder_input], axis=-1)
    outputs = create_decoder(x, decoder_params)
    return model_class(inputs=inputs, outputs=outputs, name='PELICAN', **kwargs)

def lorentznet(input_shapes : dict, ln_params : dict, mlp_params : dict, no_scalars : bool=False, edge_mask : bool = False, shared_weights:bool=False, model_class:keras.Model=keras.Model, **kwargs):
    if edge_mask:
        print("Creating LorentzNet with edge masking")
        from .lorentznetV2 import LorentzNet, LorentzNetDecoder
    else:
        from .lorentznet import LorentzNet, LorentzNetDecoder
    inputs = []
    decoder_params = ln_params["decoder"]
    ln_params = ln_params.copy()
    del ln_params["decoder"]
    features = input_shapes["scalars"][-1] if not no_scalars else None
    if shared_weights:
        nets = [LorentzNet(params=ln_params, mlp_params=mlp_params, features=features, name=f"LorentzNet-Jet")]*input_shapes["njets"]
    else:
        nets = [LorentzNet(params=ln_params, mlp_params=mlp_params, features=features, name=f"LorentzNet-Jet{idx+1:d}") for idx in range(input_shapes["njets"])]

    outputs = []

    for idx in range(input_shapes["njets"]):
        coords = keras.Input(name=f"jet{idx+1:d}_coordinates", shape=input_shapes["coordinates"])
        mask = keras.Input(name=f"jet{idx+1:d}_mask", shape=input_shapes["mask"])
        if no_scalars:
            scalars = None
            inputs.extend([coords,mask])
            outputs.append(nets[idx](coords, mask)) #(N,C)
        else:
            scalars = keras.Input(name=f"jet{idx+1:d}_scalars", shape=input_shapes["scalars"])
            inputs.extend([coords, scalars, mask])
            outputs.append(nets[idx](coords, mask, scalars=scalars)) #(N,C)

    x = _combine_jets(outputs, combiner=kwargs.pop("combiner", "concat")) #(N,C)
    if(decoder_input_shape := input_shapes.get("decoder_input", None)) is not None:
        decoder_input = keras.Input(name="decoder_input", shape=decoder_input_shape) #(N,D)
        inputs.append(decoder_input)
        x = ko.concatenate([x,decoder_input],axis=1) #(N,C+D)
    outputs = LorentzNetDecoder(decoder_params)(x) #(N,2)
    return model_class(inputs=inputs, outputs=outputs, name='LorentzNet', **kwargs)

def particlenet(input_shapes:dict, convolutions:tuple, fcs:tuple, model_class:keras.Model=keras.Model, **kwargs):
    from .particlenet import create_particle_net, create_decoder
    njets = input_shapes["njets"]
    inputs, nets = [], []
    for idx in range(njets):
        coords = keras.Input(name=f"jet{idx+1:d}_coordinates", shape=input_shapes["coordinates"])
        features = keras.Input(name=f"jet{idx+1:d}_features", shape=input_shapes["features"])
        mask = keras.Input(name=f"jet{idx+1:d}_mask", shape=input_shapes["mask"])
        inputs.extend([coords, features, mask])
        nets.append(create_particle_net(coords,features,mask,None, convolutions, fc_params=None, name=f"PN-Jet{idx+1:d}")) #(N,C)
    
    x = _combine_jets(nets, combiner=kwargs.pop("combiner", "concat")) #(N,C)
    if(decoder_input_shape := input_shapes.get("decoder_input", None)) is not None:
        decoder_input = keras.Input(name="decoder_input", shape=decoder_input_shape) #(N,D)
        inputs.append(decoder_input)
        x = ko.concatenate([x,decoder_input],axis=1) #(N,C+D)
    outputs = create_decoder(x, fcs)
    print([x.shape for x in inputs])
    return model_class(inputs=inputs, outputs=outputs, name='ParticleNet', **kwargs)
