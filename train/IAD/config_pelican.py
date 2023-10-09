FEATURES = None #Pelican only used 4 momenta
N_FEATURES = 0

msg_out = 35
agg_out = 60
layers = 5
BATCH_SIZE = 250
PL_EMBEDDING = {"dim": 20, "batchnorm":True}
PL_2TO2 = {"layers": [agg_out]*layers, "message_layers": [[msg_out]]*layers,
            "dropout": 0.1,"ir_safe": False, "factorize": True}
PL_MSG  = {"layers": [msg_out],"ir_safe": False}
PL_2TO0 = {"dim": 30, "ir_safe": False}
PL_DECODER = {"layers": [60, 30, 2],"ir_safe":False}
PL_DROPOUT_MSG = 0.1
PL_DROPOUT_OUTPUT = 0.1

LR_USE_CHAIN=True
LR_START = 0
if "MONITOR" not in globals():
    MONITOR = 'val_loss'
LR_CHAIN = [{"type": "warmup", "duration": 3, "params": {"multiplier": 1.0, "warmup_epochs": 4, "final_lr": "$<|LR_MAX|>"}},
            {"type": "cosine", "duration": 15,"params": {"T_0": 15, "T_mult": 1, "gamma": 0.3, "eta_min": 3e-05, "eta_gamma": 0.1}},
            {"type": "reduce", "duration": None, "params": {"factor": 0.2, "monitor": MONITOR, "patience": 4}}]
LR_PER_BATCH = True
ES_PATIENCE = None
LR_PATIENCE = None
