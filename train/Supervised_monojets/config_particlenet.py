FEATURES = 'all' #eta, phi, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR
N_FEATURES = 7 #eta, phi, log(pt), log(E), log(pt/pt_jet), log(E/E_jet), deltaR
_K = 16#"$<|16 if N>20 else (4 if N<20 else 8)|>"
_ACTIVATION = 'leakyrelu'
USE_CONCAT = True
PN_CONVOLUTIONS = [{"k": _K, "channels": (64,64,64), "activation": _ACTIVATION},
                    {"k": _K, "channels": (128,128,128), "activation": _ACTIVATION, "scaling": None}, 
                    {"k": _K, "channels": (256,256,256), "activation": _ACTIVATION, "scaling": None}]
PN_FCS = [{"nodes":256, "dropout":0.3, "activation": _ACTIVATION},
          {"nodes":128, "dropout":0.3, "activation": _ACTIVATION}]

BATCH_SIZE = 800

LR_USE_CHAIN=True
if "MONITOR" not in globals():
    MONITOR = 'val_loss'
LR_START = 1e-3
#LR_FINAL = 3e-3
EPOCHS = 75
LR_CHAIN = [{"type": "reduce", "duration":  None, "params": {"start_lr":LR_START, "factor":0.1, "patience": 8}}]
ES_PATIENCE = 12
ES_MIN_EPOCHS = 20
#For adamw:
WEIGHT_DECAY = 0.0001