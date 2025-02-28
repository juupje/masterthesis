FEATURES = [4]
N_FEATURES = 1
NOISE_FEATURES = None
NOISE_PARAM = (0,1)
C = 5e-3
_HIDDEN = 72
_ACTIVATION = "leakyrelu"
LN_PARAMS = {"embedding": {"dim":_HIDDEN}, "c": C, "L":6, "use_psi": True,
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

#BATCH_SIZE = 500
LR_USE_CHAIN=True
LR_MAX = 1e-4
LR_START = "$<|LR_MAX|>"
EPOCHS = 35
if "MONITOR" not in globals():
    MONITOR = 'loss'
LR_CHAIN = [#{"type": "warmup", "duration": 3, "params": {"multiplier": 1.0, "warmup_epochs": 4, "final_lr": "$<|LR_MAX|>"}},
            {"type": "cosine", "duration": 15,"params": {"T_0": 15, "T_mult": 1, "gamma": 0.3, "eta_min": 3e-05, "eta_gamma": 0.1}},
            {"type": "reduce", "duration": None, "params": {"factor": 0.2, "monitor": MONITOR, "patience": 4}}]
ES_PATIENCE = None
#ES_MIN_DELTA = 0.0001
#ES_MIN_EPOCHS = 17