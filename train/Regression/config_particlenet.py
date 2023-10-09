FEATURES = [0,1,2,3,6,7,8] #eta, phi, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR
N_FEATURES = 7 #eta, phi, log(pt), log(E), log(pt/pt_jet), log(E/E_jet), deltaR
_K = "$<|16 if N>20 else (4 if N<20 else 8)|>"
_ACTIVATION = 'leakyrelu'
PN_CONVOLUTIONS = [{"k": _K, "channels": (64,64,64), "activation": _ACTIVATION},
                    {"k": _K, "channels": (128,128,128), "activation": _ACTIVATION, "scaling": None}, 
                    {"k": _K, "channels": (256,256,256), "activation": _ACTIVATION, "scaling": None}]
PN_FCS = [{"nodes":256, "dropout":0.1, "activation": _ACTIVATION},
          {"nodes":128, "dropout":0.1, "activation": _ACTIVATION},
          {"nodes":1, "activation": 'linear'}]

BATCH_SIZE = 800
EPOCHS = 35
LR_USE_CHAIN=True
LR_START = 3e-4
LR_FINAL = 3e-3
'''
LR_CHAIN = [{"type": "warmup", "duration":  8, "params": {"start_lr":LR_START, "multiplier":1.0, "warmup_epochs":8, "final_lr":LR_FINAL}},
            {"type": "cooldown", "duration":  8, "params": {"multiplier":1.0, "cooldown_epochs":8, "final_lr":LR_START}},
            {"type": "cooldown", "duration":  4, "params": {"multiplier":1.0, "cooldown_epochs":4, "final_lr": 1E-7}}]
'''
LR_CHAIN = [{"type": "cosine", "duration": 15, "params": {"T_0":15, "T_mult":2, "eta_min":1e-4}},
            #{"type": "constant", "duration": 5, "params": {}},
            #{"type": "cosine", "duration": 6, "params": {"start_lr": LR_START*0.75, "T_0":6, "T_mult":2, "eta_min":5e-5}},
            {"type": "step", "duration": None, "params": {"factor": 0.2, "T_wait": 6}}]
ES_PATIENCE = 6
ES_MIN_EPOCHS = 6

#For adamw:
WEIGHT_DECAY = 0.0001