#FEATURES = None
N_FEATURES = "$<|len(FEATURES)|>"
NOISE_FEATURES = None
NOISE_PARAM = (0,1)
C = 5e-3
_HIDDEN = 72
_ACTIVATION = "leakyrelu"
LN_PARAMS = {"embedding": {"dim":_HIDDEN}, "c": "$<|C|>", "L":6, "use_psi": True,
            "decoder":[{"N": 2*_HIDDEN, "activation": _ACTIVATION, "dropout": 0.1},
                       {"N": _HIDDEN, "activation": _ACTIVATION, "dropout": 0.1},
                       {"N": _HIDDEN//2, "activation": _ACTIVATION, "dropout": 0.1}]}
LN_PHI_E = ({"N": _HIDDEN, "activation": _ACTIVATION, "batchnorm": True},
        {"N": _HIDDEN})
LN_PHI_X = ({"N": _HIDDEN, "activation": _ACTIVATION},
        {"N": 1, "bias":False})
LN_PHI_H = ({"N": _HIDDEN, "activation": _ACTIVATION, "batchnorm": True},
        {"N": _HIDDEN})
LN_PHI_M = ({"N": 1, "activation": "sigmoid"},)

BATCH_SIZE = 512
LR_USE_CHAIN=True
LR_MAX = 1e-3
LR_START = "$<|LR_MAX|>"
if "MONITOR" not in globals():
    MONITOR = 'val_loss'
EPOCHS = 75
LR_CHAIN = [{"type": "reduce", "duration":  None, "params": {"start_lr":LR_START, "factor":0.1, "patience": 8, "monitor": MONITOR}}]

ES_PATIENCE = 8
ES_MIN_EPOCHS = 17