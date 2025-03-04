FEATURES = None
N_FEATURES = 0
NOISE_FEATURES = None
NOISE_PARAM = (0,1)

_HIDDEN = 72
_ACTIVATION = "relu"
LN_PARAMS = {"embedding": {"dim":_HIDDEN}, "c": 5e-3, "L":6, "use_psi": True,
            "decoder":[{"N": _HIDDEN, "activation": _ACTIVATION, "dropout": 0.2},
                       {"N": 1, "activation": "linear"}]}
LN_PHI_E = ({"N": _HIDDEN, "activation": _ACTIVATION, "batchnorm": True},
        {"N": _HIDDEN})
LN_PHI_X = ({"N": _HIDDEN, "activation": _ACTIVATION},
        {"N": 1, "bias":False})
LN_PHI_H = ({"N": _HIDDEN, "activation": _ACTIVATION, "batchnorm": True},
        {"N": _HIDDEN})
LN_PHI_M = ({"N": 1, "activation": "sigmoid"},)

#BATCH_SIZE = 256
LR_USE_CHAIN=True
LR_MAX = 1e-4
LR_START = "$<|LR_MAX|>"
EPOCHS = 35
if "MONITOR" not in globals():
    MONITOR = 'val_loss'
LR_CHAIN = [#{"type": "warmup", "duration": 3, "params": {"multiplier": 1.0, "warmup_epochs": 4, "final_lr": "$<|LR_MAX|>"}},
            {"type": "cosine", "duration": 15,"params": {"T_0": 15, "T_mult": 1, "gamma": 0.3, "eta_min": 3e-05, "eta_gamma": 0.1}},
            {"type": "reduce", "duration": None, "params": {"factor": 0.2, "monitor": MONITOR, "patience": 4}}]
ES_PATIENCE = None
#ES_MIN_EPOCHS = 6