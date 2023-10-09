import tensorflow as tf
from tensorflow import keras
from models import masked_batch_normalization as mbn, pelican
from models import ScalingLayer
import utils
from utils import hadamard
from metrics.SingleClassLoss import SingleClassLoss
from metrics.ROC import ROCMetric
import numpy as np

class WeakModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sb_loss = SingleClassLoss(class_label=0, name='background_region_loss')
        self.sr_loss = SingleClassLoss(class_label=1, name='signal_region_loss')
        self.bg_loss = SingleClassLoss(class_label=0, name='bg_loss')
        self.sn_loss = SingleClassLoss(class_label=1, name='sn_loss')
        self.extra_metrics = [self.sb_loss, self.sr_loss, self.bg_loss, self.sn_loss]
        #self.roc_metric = ROCMetric(name='roc_metric', tpr_values=[0.2,0.4], fpr_values=[1e-4])

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        if(isinstance(y, (tuple, list))):
            y_weak, y_true = y
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y_weak, y_pred, regularization_losses=self.losses, sample_weight=sample_weight)
        # Update the metrics.
        self.compiled_metrics.update_state(y_weak, y_pred)
        self.sb_loss.update_state(y_weak, y_pred)
        self.sr_loss.update_state(y_weak, y_pred)
        self.bg_loss.update_state(y_true, y_pred)
        self.sn_loss.update_state(y_true, y_pred)
        #self.roc_metric.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        results = {m.name: m.result() for m in self.metrics}
        results.update({m.name: m.result() for m in self.extra_metrics})
        return results

class SupervisedModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_loss = SingleClassLoss(class_label=0, name='bg_loss')
        self.sn_loss = SingleClassLoss(class_label=1, name='sn_loss')
        self.extra_metrics = [self.bg_loss, self.sn_loss]
        #self.roc_metric = ROCMetric(name='roc_metric', tpr_values=[0.2,0.4], fpr_values=[1e-4])

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y_true = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses, sample_weight=sample_weight)
        # Update the metrics.
        self.compiled_metrics.update_state(y_true, y_pred)
        self.bg_loss.update_state(y_true, y_pred)
        self.sn_loss.update_state(y_true, y_pred)
        #self.roc_metric.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        results = {m.name: m.result() for m in self.metrics}
        results.update({m.name: m.result() for m in self.extra_metrics})
        return results

def create_single_jet_model(config:dict, model_class=WeakModel) -> tf.keras.Model:
    if(config["MODEL"].lower() == "particlenet"):
        if(config.get("USE_CONCAT", False)):
            from models import particlenet_concat
            model = particlenet_concat.particle_net_concat(
                                    input_shapes=dict(coordinates=(config["N"],2), features=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1)),
                                    convolutions = config["PN_CONVOLUTIONS"],
                                    fcs= config["PN_FCS"], model_class=model_class)
        else:
            from models import particlenet
            model = particlenet.particle_net(
                        input_shapes=dict(coordinates=(config["N"],2), features=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1)),
                        convolutions = config["PN_CONVOLUTIONS"],
                        fcs= config["PN_FCS"], model_class=model_class)
    elif(config["MODEL"].lower() == "lorentznet"):
        from models import lorentznet
        if(config["N_FEATURES"]==0): #Here we remove the first scalar input all-together
            model = lorentznet.lorentz_net(input_shapes=dict(coordinates=(config["N"],4), mask=(config["N"],1)),
                        ln_params=config["LN_PARAMS"],
                        mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]),
                        no_scalars=True, model_class=model_class)
        else: #here, we just feed in zeros as the scalar variables 
            model = lorentznet.lorentz_net(input_shapes=dict(coordinates=(config["N"],4), scalars=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1)),
                        ln_params=config["LN_PARAMS"],
                        mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]), model_class=model_class)
    elif(config["MODEL"].lower() == "pelican"):
        params = {"embedding": config["PL_EMBEDDING"], "2to2": config["PL_2TO2"], "message": config["PL_MSG"],
                "dropout_msg": config["PL_DROPOUT_MSG"], "2to0": config["PL_2TO0"], "dropout_out": config["PL_DROPOUT_OUTPUT"], "decoder": config["PL_DECODER"]}
        model = pelican.pelican(input_shapes={"coordinates": (config["N"],4), "mask": (config["N"],1)}, params=params, model_class=model_class)
    else:
        raise ValueError("Unknown model " + config["MODEL"])
    print(f"Created {config['MODEL']}:\n"+"\n".join(utils.extract_params_from_summary(model)))
    if(config["OPTIMIZER"].lower()=="adam"):
        optimizer = keras.optimizers.Adam(learning_rate=config["LR_START"])
    elif(config["OPTIMIZER"].lower()=="adamw"):
        optimizer = keras.optimizers.experimental.AdamW(learning_rate=config["LR_START"], weight_decay=config.get("WEIGHT_DECAY", 0.01))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    return model

def create_dijet_model(config:dict, model_class=WeakModel) -> tf.keras.Model:
    from models import jets
    if(config["MODEL"].lower() == "particlenet"):
        model = jets.particlenet(
                    input_shapes=dict(njets=2,coordinates=(config["N"],2), features=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1)),
                    convolutions = config["PN_CONVOLUTIONS"],
                    fcs= config["PN_FCS"], model_class=model_class)
    elif(config["MODEL"].lower() == "lorentznet"):
        if(config["N_FEATURES"]==0): #Here we remove the first scalar input all-together
            model = jets.lorentznet(input_shapes=dict(njets=2, coordinates=(config["N"],4), mask=(config["N"],1)),
                        ln_params=config["LN_PARAMS"],
                        mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]),
                        no_scalars=True, model_class=model_class)
        else: #here, we just feed in zeros as the scalar variables 
            model = jets.lorentznet(input_shapes=dict(njets=2,coordinates=(config["N"],4), scalars=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1)),
                        ln_params=config["LN_PARAMS"],
                        mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]),
                        model_class=model_class)
    elif(config["MODEL"].lower() == "pelican"):
        params = {"embedding": config["PL_EMBEDDING"], "2to2": config["PL_2TO2"], "message": config["PL_MSG"],
                "dropout_msg": config["PL_DROPOUT_MSG"], "2to0": config["PL_2TO0"], "dropout_out": config["PL_DROPOUT_OUTPUT"], "decoder": config["PL_DECODER"]}
        model = jets.pelican(input_shapes={"njets":2, "coordinates": (config["N"],4), "mask": (config["N"],1)}, params=params,
                             model_class=model_class)
    else:
        raise ValueError("Unknown model " + config["MODEL"])
    print(f"Created {config['MODEL']}:\n"+"\n".join(utils.extract_params_from_summary(model)))
    if(config["OPTIMIZER"].lower()=="adam"):
        optimizer = keras.optimizers.Adam(learning_rate=config["LR_START"])
    elif(config["OPTIMIZER"].lower()=="adamw"):
        optimizer = keras.optimizers.experimental.AdamW(learning_rate=config["LR_START"], weight_decay=config.get("WEIGHT_DECAY", 0.01))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    return model

def create_regression_model(config:dict) -> tf.keras.Model:
    from . import regression
    if(config["MODEL"].lower() == "particlenet"):
        model = regression.particlenet(
                    input_shapes=dict(njets=2,coordinates=(config["N"],2), features=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1)),
                    convolutions = config["PN_CONVOLUTIONS"],
                    fcs=config["PN_FCS"])
    elif(config["MODEL"].lower() == "lorentznet"):
        if(config["N_FEATURES"]==0): #Here we remove the first scalar input all-together
            model = regression.lorentznet(input_shapes=dict(njets=2, coordinates=(config["N"],4), mask=(config["N"],1)),
                        ln_params=config["LN_PARAMS"],
                        mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]),
                        no_scalars=True)
        else: #here, we just feed in zeros as the scalar variables 
            model = regression.lorentznet(input_shapes=dict(njets=2,coordinates=(config["N"],4), scalars=(config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0)), mask=(config["N"],1)),
                        ln_params=config["LN_PARAMS"],
                        mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]))
    elif(config["MODEL"].lower() == "pelican"):
        params = {"embedding": config["PL_EMBEDDING"], "2to2": config["PL_2TO2"], "message": config["PL_MSG"],
                "dropout_msg": config["PL_DROPOUT_MSG"], "2to0": config["PL_2TO0"], "dropout_out": config["PL_DROPOUT_OUTPUT"], "decoder": config["PL_DECODER"]}
        model = regression.pelican(input_shapes={"njets":2, "coordinates": (config["N"],4), "mask": (config["N"],1)}, params=params)
    else:
        raise ValueError("Unknown model " + config["MODEL"])
    print(f"Created {config['MODEL']}:\n"+"\n".join(utils.extract_params_from_summary(model)))
    if(config["OPTIMIZER"].lower()=="adam"):
        optimizer = keras.optimizers.Adam(learning_rate=config["LR_START"])
    elif(config["OPTIMIZER"].lower()=="adamw"):
        optimizer = keras.optimizers.experimental.AdamW(learning_rate=config["LR_START"], weight_decay=config.get("WEIGHT_DECAY", 0.01))
    model.compile(optimizer=optimizer, loss='mse')
    return model

def load_model(path:str, modeltype:str) -> tf.keras.Model:
    customobjects = {"PelicanEmbedding":pelican.PelicanEmbedding, "Eq2to2": pelican.Eq2to2,
                        "Eq2to0": pelican.Eq2to0, "MaskedBatchNormalization": mbn.MaskedBatchNormalization} if modeltype.upper()=="PELICAN" else {}
    customobjects["SingleClassLoss"] = SingleClassLoss
    customobjects["ScalingLayer"] = ScalingLayer.ScalingLayer
    customobjects["WeakModel"] = WeakModel
    customobjects["SupervisedModel"] = SupervisedModel
    customobjects["hadamard_weights"] = hadamard.hadamard_weights
    return tf.keras.models.load_model(path, custom_objects=customobjects)