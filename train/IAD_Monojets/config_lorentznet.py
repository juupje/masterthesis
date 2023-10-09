FEATURES = [2,4]
N_FEATURES = 2
NOISE_FEATURES = None
NOISE_PARAM = (0,1)
C = 5e-3
_HIDDEN = 72
_ACTIVATION = "leakyrelu"
LN_PARAMS = {"embedding": {"dim":_HIDDEN}, "c": "$<|C|>", "L":6, "use_psi": True,
            "decoder":[{"N": 2*_HIDDEN, "activation": _ACTIVATION, "dropout": 0.1},
                       {"N": _HIDDEN, "activation": _ACTIVATION, "dropout": 0.1}]}
LN_PHI_E = ({"N": _HIDDEN, "activation": _ACTIVATION, "batchnorm": True},
        {"N": _HIDDEN})
LN_PHI_X = ({"N": _HIDDEN, "activation": _ACTIVATION},
        {"N": 1, "bias":False})
LN_PHI_H = ({"N": _HIDDEN, "activation": _ACTIVATION, "batchnorm": True},
        {"N": _HIDDEN})
LN_PHI_M = ({"N": 1, "activation": "sigmoid"},)

BATCH_SIZE = 512
LR_USE_CHAIN=True
MONITOR = 'loss'
'''
EPOCHS = 35
LR_MAX = 1e-3
LR_START = "$<|LR_MAX|>"
LR_CHAIN = [#{"type": "warmup", "duration": 3, "params": {"multiplier": 1.0, "warmup_epochs": 4, "final_lr": "$<|LR_MAX|>"}},
            {"type": "cosine", "duration": 15,"params": {"T_0": 15, "T_mult": 1, "gamma": 0.3, "eta_min": 3e-05, "eta_gamma": 0.1}},
            {"type": "step", "duration": None, "params": {"factor": 0.2, "monitor": MONITOR, "patience": 6}}]

ES_PATIENCE = 8
ES_MIN_EPOCHS = 17
'''
LR_START = 1e-3
#LR_FINAL = 3e-3
EPOCHS = 70
LR_CHAIN = [{"type": "reduce", "duration":  None, "params": {"start_lr":LR_START, "factor":0.1, "min_delta": 0.0001,"monitor": MONITOR, "patience": 8}}]
ES_PATIENCE = 12
ES_MIN_EPOCHS = 20
ES_MIN_DELTA=1e-4