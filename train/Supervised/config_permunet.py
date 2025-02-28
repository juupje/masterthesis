FEATURES = [0,1,2,3,6,7,8] #eta, phi, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR
N_FEATURES = 7 #eta, phi, log(pt), log(E), log(pt/pt_jet), log(E/E_jet), deltaR
_ACTIVATION = 'relu'
PERMUNET_PARAMS = {"embedding": {"channels": 64, "activation": _ACTIVATION},
                   "layers": [{"channels": 64, "activation": _ACTIVATION, "update_scale": 0.1, "dropout": 0.2},
                              {"channels": 64, "activation": _ACTIVATION, "update_scale": 0.1, "dropout": 0.2},
                              {"channels": 64, "activation": _ACTIVATION, "update_scale": 0.1, "dropout": 0.2}]}
PERMUNET_DECODER = [{"nodes":128, "dropout":0.3, "activation": _ACTIVATION}]

BATCH_SIZE = 512

OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.0001
LR_USE_CHAIN=True
if "MONITOR" not in globals():
    MONITOR = 'val_loss'
EPOCHS = 20
LR_START = 3e-4
LR_FINAL = 3e-3
LR_CHAIN = [{"type": "warmup", "duration":  8, "params": {"start_lr":LR_START, "multiplier":1.0, "warmup_epochs":8, "final_lr":LR_FINAL}},
            {"type": "cooldown", "duration":  8, "params": {"multiplier":1.0, "cooldown_epochs":8, "final_lr":LR_START}},
            {"type": "cooldown", "duration":  4, "params": {"multiplier":1.0, "cooldown_epochs":4, "final_lr": 5e-7}}]
ES_PATIENCE = None