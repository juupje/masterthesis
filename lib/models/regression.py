import tensorflow as tf
from utils.configs import marshal

def pelican(input_shapes : dict, params : dict, model_class:tf.keras.Model=tf.keras.Model) -> tf.keras.Model:
    from .pelican import create_pelican, create_decoder, default_decoder
    decoder_params = marshal(params.get("decoder"), default_decoder, "decoder",quiet=True)
    assert decoder_params["layers"][-1]==1, "A regression model should have one output"
    params = params.copy()
    del params["decoder"]
    jets,nets = [], []
    for idx in range(input_shapes["njets"]):   
        coords = tf.keras.Input(name=f"jet{idx+1:d}_coordinates", shape=input_shapes["coordinates"])
        mask = tf.keras.Input(name=f"jet{idx+1:d}_mask", shape=input_shapes["mask"])
        jets.extend([coords,mask])
        nets.append(create_pelican(coords,mask, params=params, name=f"PELICAN-Jet{idx+1:d}")) #(N,C)

    x = tf.concat(nets,axis=1)
    outputs = create_decoder(x, decoder_params)
    return model_class(inputs=jets, outputs=outputs, name='PELICAN')

def lorentznet(input_shapes : dict, ln_params : dict, mlp_params : dict, no_scalars : bool=False, model_class:tf.keras.Model=tf.keras.Model):
    from .lorentznet import create_lorentz_net, create_decoder
    jets, nets = [], []
    decoder_params = ln_params["decoder"]
    assert decoder_params[-1]["N"]==1, "A regression model should have one output"
    ln_params = ln_params.copy()
    del ln_params["decoder"]
    for idx in range(input_shapes["njets"]):
        coords = tf.keras.Input(name=f"jet{idx+1:d}_coordinates", shape=input_shapes["coordinates"])
        mask = tf.keras.Input(name=f"jet{idx+1:d}_mask", shape=input_shapes["mask"])
        if no_scalars:
            scalars = None
            jets.extend([coords,mask])
        else:
            scalars = tf.keras.Input(name=f"jet{idx+1:d}_scalars", shape=input_shapes["scalars"])
            jets.extend([coords, scalars, mask])
        nets.append(create_lorentz_net(coords,scalars,mask, params=ln_params, mlp_params=mlp_params, name=f"LorentzNet-Jet{idx+1:d}")) #(N,C)
    
    x = tf.concat(nets,axis=1) #(N,J*C)
    outputs = create_decoder(x, decoder_params, softmax=False)
    return model_class(inputs=jets, outputs=outputs, name='LorentzNet')

def particlenet(input_shapes:dict, convolutions:tuple, fcs:tuple, model_class:tf.keras.Model=tf.keras.Model):
    from .particlenet import create_particle_net, create_decoder
    njets = input_shapes["njets"]
    assert fcs[-1]["nodes"]==1, "A regression model should have one output"
    jets, nets = [], []
    for idx in range(njets):
        coords = tf.keras.Input(name=f"jet{idx+1:d}_coordinates", shape=input_shapes["coordinates"])
        features = tf.keras.Input(name=f"jet{idx+1:d}_features", shape=input_shapes["features"])
        mask = tf.keras.Input(name=f"jet{idx+1:d}_mask", shape=input_shapes["mask"])
        jets.extend([coords, features, mask])
        nets.append(create_particle_net(coords,features,mask, convolutions, fc_params=None, name=f"PN-Jet{idx+1:d}")) #(N,C)
    
    x = tf.concat(nets,axis=1) #(N,J*C), J=#jets
    outputs = create_decoder(x, fcs, softmax=False)
    return model_class(inputs=jets, outputs=outputs, name='ParticleNet')
