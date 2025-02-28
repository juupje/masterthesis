import keras
import utils
from utils import hadamard
from .CustomModels import WeakModel, SupervisedModel
from metrics import SingleClassLoss
try:
    from .CustomOptimizers import BatchedAdamW
except:
    pass
import logging
logger = logging.getLogger(__name__)

def _get_optimizer(config):
    update_steps = config.get("UPDATE_STEPS", 1)
    if update_steps == 1: update_steps = None
    optimizer = config["OPTIMIZER"].lower()
    if(optimizer=="adam"):
        optimizer = keras.optimizers.Adam(learning_rate=float(config["LR_START"]), gradient_accumulation_steps=update_steps)
    elif(optimizer=="adamw"):
        optimizer = keras.optimizers.AdamW(learning_rate=float(config["LR_START"]), weight_decay=config.get("WEIGHT_DECAY", 0.01), gradient_accumulation_steps=update_steps)
    return optimizer

def create_single_jet_model(config:dict, model_class=WeakModel) -> keras.Model:
    jit_compile = True
    if(dec_inputs := config.get("DECODER_INPUTS", None)):
        from DataLoaders import DecoderFeatures
        dec_input = {"decoder_input": (sum([DecoderFeatures.get(dec).dimension for dec in dec_inputs]),)}
        logger.info("Decoder input shape", dec_input["decoder_input"])
    else:
        dec_input = {}
    if(config["MODEL"].lower() == "permunet"):
        from models import permunet
        model = permunet.permunet(input_shapes=dict(coordinates=(config["N"],2), features=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1), **dec_input),
                        params = config["PERMUNET_PARAMS"], decoder= config["PERMUNET_DECODER"], model_class=model_class, **config.get("MODEL_ARGS", {}))
    elif(config["MODEL"].lower() == "particlenet"):
        if(config.get("USE_CONCAT", False)):
            from models import particlenet_concat
            model = particlenet_concat.particle_net_concat(
                                    input_shapes=dict(coordinates=(config["N"],2), features=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1), **dec_input),
                                    convolutions = config["PN_CONVOLUTIONS"],
                                    fcs= config["PN_FCS"], model_class=model_class, **config.get("MODEL_ARGS", {}))
        else:
            from models import particlenet
            model = particlenet.particle_net(
                        input_shapes=dict(coordinates=(config["N"],2), features=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1), **dec_input),
                        convolutions = config["PN_CONVOLUTIONS"],
                        fcs= config["PN_FCS"], model_class=model_class, **config.get("MODEL_ARGS", {}))
    elif(config["MODEL"].lower().startswith("lorentznet")):
        from DataLoaders import DecoderFeatures
        if(config["MODEL"].lower() == "lorentznetv2"):
            jit_compile = False
            from . import lorentznetV2 as lorentznet
        else:
            from . import lorentznet
        input_shapes = dict(coordinates=(config["N"],4), mask=(config["N"],1), **dec_input)
        if(config["N_FEATURES"]!=0): #add scalar features
            input_shapes["scalars"] = (config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0))
        model = lorentznet.lorentz_net(input_shapes=input_shapes, ln_params=config["LN_PARAMS"],
                    mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]),
                    no_scalars=config["N_FEATURES"]==0, model_class=model_class, **config.get("MODEL_ARGS", {}))
    elif(config["MODEL"].lower() == "pelican"):
        from . import pelican
        params = {"embedding": config["PL_EMBEDDING"], "2to2": config["PL_2TO2"], "message": config["PL_MSG"],
                "dropout_msg": config["PL_DROPOUT_MSG"], "2to0": config["PL_2TO0"], "dropout_out": config["PL_DROPOUT_OUTPUT"], "decoder": config["PL_DECODER"]}
        model = pelican.pelican(input_shapes={"coordinates": (config["N"],4), "mask": (config["N"],1), **dec_input}, params=params,
                                model_class=model_class, **config.get("MODEL_ARGS", {}))
    else:
        raise ValueError("Unknown model " + config["MODEL"])
    logger.info(f"Created {config['MODEL']}:\n"+"\n".join(utils.formatting.extract_params_from_summary(model)))
    logger.info(f"Using JIT compilation: {jit_compile}")
    model.compile(optimizer=_get_optimizer(config), loss='categorical_crossentropy', metrics=["accuracy"], jit_compile=jit_compile)
    return model

def create_dijet_model(config:dict, model_class=WeakModel) -> keras.Model:
    from models import jets
    jit_compile=True
    if(dec_inputs := config.get("DECODER_INPUTS", None)):
        from DataLoaders import DecoderFeatures
        dec_input = {"decoder_input": (sum([DecoderFeatures.get(dec).dimension for dec in dec_inputs]),)}
        logger.info(f"Decoder input shape {dec_input['decoder_input']}")
    else:
        dec_input = {}
    if(config["MODEL"].lower() == "particlenet"):
        model = jets.particlenet(
                    input_shapes=dict(njets=2,coordinates=(config["N"],2), features=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1), **dec_input),
                    convolutions = config["PN_CONVOLUTIONS"],
                    fcs= config["PN_FCS"], model_class=model_class, **config.get("MODEL_ARGS", {}))
    elif(config["MODEL"].lower().startswith("lorentznet")):
        v2 = config["MODEL"].lower().endswith("v2")
        jit_compile = jit_compile and not v2
        input_shapes = dict(njets=2, coordinates=(config["N"],4), mask=(config["N"],1), **dec_input)
        if(config["N_FEATURES"]!=0): #Add scalar features
            input_shapes["scalars"] = (config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0))
        model = jets.lorentznet(input_shapes=input_shapes, ln_params=config["LN_PARAMS"],
                    mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]),
                    no_scalars=config["N_FEATURES"]==0, edge_mask=v2, shared_weights=config.get("SHARED_WEIGHTS", False), model_class=model_class, **config.get("MODEL_ARGS", {}))
    elif(config["MODEL"].lower() == "pelican"):
        params = {"embedding": config["PL_EMBEDDING"], "2to2": config["PL_2TO2"], "message": config["PL_MSG"],
                "dropout_msg": config["PL_DROPOUT_MSG"], "2to0": config["PL_2TO0"], "dropout_out": config["PL_DROPOUT_OUTPUT"], "decoder": config["PL_DECODER"]}
        model = jets.pelican(input_shapes={"njets":2, "coordinates": (config["N"],4), "mask": (config["N"],1), **dec_input}, params=params,
                             model_class=model_class, **config.get("MODEL_ARGS", {}))
    else:
        raise ValueError("Unknown model " + config["MODEL"])
    logger.info(f"Created {config['MODEL']}:\n"+"\n".join(utils.formatting.extract_params_from_summary(model)))
    logger.info(f"Using JIT compilation: {jit_compile}")
    model.compile(optimizer=_get_optimizer(config), loss='categorical_crossentropy', metrics=["accuracy"], jit_compile=jit_compile)
    return model

def create_regression_model(config:dict) -> keras.Model:
    from . import regression
    jit_compile = True
    if(dec_inputs := config.get("DECODER_INPUTS", None)):
        from DataLoaders import DecoderFeatures
        dec_input = {"decoder_input": (sum([DecoderFeatures.get(dec).dimension for dec in dec_inputs]),)}
        logger.info(f"Decoder input shape {dec_input['decoder_input']}")
    else:
        dec_input = {}
    if(config["MODEL"].lower() == "particlenet"):
        model = regression.particlenet(
                    input_shapes=dict(njets=2,coordinates=(config["N"],2), features=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1), **dec_input),
                    convolutions = config["PN_CONVOLUTIONS"],
                    fcs=config["PN_FCS"])
    elif(config["MODEL"].lower().startswith("lorentznet")):
        v2 = config["MODEL"].lower().endswith("v2")
        jit_compile = jit_compile and not v2
        dijet = not config.get("MERGE_JETS", False)
        if(config["N_FEATURES"]==0): #Here we remove the first scalar input all-together
            model = regression.lorentznet(input_shapes=dict(njets=2 if dijet else 1, coordinates=(config["N"],4), mask=(config["N"],1), **dec_input),
                        ln_params=config["LN_PARAMS"],
                        mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]),
                        no_scalars=True, edge_mask=v2)
        else:
            model = regression.lorentznet(input_shapes=dict(njets=2 if dijet else 1,coordinates=(config["N"],4), scalars=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1), **dec_input),
                        ln_params=config["LN_PARAMS"],
                        mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]),
                        edge_mask=v2)
    elif(config["MODEL"].lower() == "pelican"):
        params = {"embedding": config["PL_EMBEDDING"], "2to2": config["PL_2TO2"], "message": config["PL_MSG"],
                "dropout_msg": config["PL_DROPOUT_MSG"], "2to0": config["PL_2TO0"], "dropout_out": config["PL_DROPOUT_OUTPUT"], "decoder": config["PL_DECODER"]}
        model = regression.pelican(input_shapes={"njets":2, "coordinates": (config["N"],4), "mask": (config["N"],1), **dec_input}, params=params)
    else:
        raise ValueError("Unknown model " + config["MODEL"])
    logger.info(f"Created {config['MODEL']}:\n"+"\n".join(utils.formatting.extract_params_from_summary(model)))
    logger.info(f"Using JIT compilation: {jit_compile}")
    model.compile(optimizer=_get_optimizer(config), loss='mse', jit_compile=jit_compile)
    return model

def load_model(path:str, modeltype:str, compile:bool=True) -> keras.Model:
    customobjects = {}
    customobjects["SingleClassLoss"] = SingleClassLoss
    customobjects["WeakModel"] = WeakModel
    customobjects["SupervisedModel"] = SupervisedModel
    customobjects["BatchedAdamW"] = BatchedAdamW
    customobjects["hadamard_weights"] = hadamard.hadamard_weights

    if modeltype.upper()=="PELICAN":
        from . import masked_batch_normalization as mbn, pelican
        customobjects.update({"PelicanEmbedding":pelican.PelicanEmbedding, "Eq2to2": pelican.Eq2to2,
                        "Eq2to0": pelican.Eq2to0, "MaskedBatchNormalization": mbn.MaskedBatchNormalization})
    elif modeltype.upper() == "PARTICLENET":
        from . import ScalingLayer
        from .feature_layers import KNN_Features
        customobjects["ScalingLayer"] = ScalingLayer.ScalingLayer
        customobjects["KNN_Features"] = KNN_Features
        customobjects["KNN_Features"] = KNN_Features
    elif modeltype.upper().startswith("LORENTZNET"):
        from .feature_layers import AdjacencyMatrix
        customobjects["AdjacencyMatrix"] = AdjacencyMatrix
    return keras.models.load_model(path, custom_objects=customobjects, compile=compile)