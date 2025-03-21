FEATURES = None
N_FEATURES = "$<|len(FEATURES or [])|>"
NOISE_FEATURES = None
NOISE_PARAM = (0,1)

_HIDDEN = 72
_ACTIVATION = "relu"
LN_PARAMS = {"embedding": {"dim":_HIDDEN}, "c": 5e-3, "L":6, "use_psi": True,
            "decoder":[{"N": _HIDDEN, "activation": _ACTIVATION, "dropout": 0.2}]}
LN_PHI_E = ({"N": _HIDDEN, "activation": _ACTIVATION, "batchnorm": True},
        {"N": _HIDDEN})
LN_PHI_X = ({"N": _HIDDEN, "activation": _ACTIVATION},
        {"N": 1, "bias":False})
LN_PHI_H = ({"N": _HIDDEN, "activation": _ACTIVATION, "batchnorm": True},
        {"N": _HIDDEN})
LN_PHI_M = ({"N": 1, "activation": "sigmoid"},)

OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.01
BATCH_SIZE = 128
LR_USE_CHAIN=True
LR_MAX = 1e-3
LR_START = 0
if "MONITOR" not in globals():
    MONITOR = 'val_loss'
EPOCHS = 35
LR_CHAIN = [{"type": "warmup", "duration":  4, "params": {"multiplier":1.0, "warmup_epochs":4, "final_lr":LR_MAX}},
            {"type": "cosine", "duration": 28, "params": {"T_0":4, "T_mult":2, "eta_min":1e-5}},
            {"type": "expdecay", "duration": 3, "params": {"gamma": 0.5}}]
ES_PATIENCE = None