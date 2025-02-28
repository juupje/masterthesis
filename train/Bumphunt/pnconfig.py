#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "CWoLa"
MODELNAME = "ParticleNet" # used in plots
RUN_ID = None

def create_grid(name_format, values, **kwargs):
    from itertools import product
    names = list(values.keys())
    vals = [values[name] for name in names]
    grid = list(product(*vals))
    configs = []
    for combination in grid:
        conf = {names[i]:combination[i] for i in range(len(combination))}
        conf["name"] = name_format(**conf)
        for key in kwargs:
            conf[key] = kwargs[key](**conf)
        configs.append(conf)
    return configs 

WORK_DIR = f"{HOME}/thesis"
DATA_DIR = os.path.join(os.getenv("DATA_DIR"), "lhco")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = f"{WORK_DIR}/outputs/{JOBNAME}"
RUN_ID_FILE = f"{WORK_DIR}/run_id.json"
CHECKPOINT_DIR = "model-checkpoints"
CHECKPOINT_FREQ = None

# ===== MODEL ======
MODEL = MODELNAME #used to load the model
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.01
N=30

EXTRA_CONFIGS = []
if(MODEL=="PELICAN"):
    EXTRA_CONFIGS.append("config_pelican.py")
elif(MODEL=="LorentzNet"):
    EXTRA_CONFIGS.append("config_lorentznet.py")
elif(MODEL=="ParticleNet"):
    EXTRA_CONFIGS.append("config_particlenet.py")

# ===== TRAINING =====
CHANGE_SEED = True #for batch processing
SEED = 4
DATA_SEED = 2
#BATCH_SIZE = 128 set by model specific config
VERBOSITY=2
PERFORM_TEST = True

#Background sizes
factor = 1
BG_SLICE = 468_000
SN_SLICE = 1000
N_BG_TRAIN = -1
N_SN_TRAIN   = -1

VAL_RATIO = 1.0
N_BG_VAL = int(VAL_RATIO*N_BG_TRAIN) if N_BG_TRAIN > 0 else -1
N_SN_VAL   = int(VAL_RATIO*N_SN_TRAIN) if N_SN_TRAIN > 0 else -1

SIGNAL_RATIO = N_SN_TRAIN/(N_BG_TRAIN+N_SN_TRAIN) if not(N_BG_TRAIN==-1 and N_SN_TRAIN==-1) else SN_SLICE/(SN_SLICE+BG_SLICE)
VAL_SIGNAL_RATIO = N_SN_VAL/(N_BG_VAL+N_SN_VAL) if not(N_BG_TRAIN==-1 and N_SN_TRAIN==-1) else None

N_TEST = 100_000
# Define the bands
SB_LEFT  = (2900, 3300)
SR       = (3300, 3700)
SB_RIGHT = (3700, 4100)

OVERSAMPLING = 'repeat_strict'

# LR scheduler
MONITOR = 'val_loss'
LR_PER_BATCH = True

#Data files
PREPROCESSED_TRAIN_DATA_BG = f"new/N100-bg.h5"
PREPROCESSED_TRAIN_DATA_SN = f"new/N100-sn.h5"
PREPROCESSED_TEST_DATA_BG  = f"new/N100-bg-SR-small.h5"
PREPROCESSED_TEST_DATA_SN  = f"new/N100-sn-test.h5"

# ====== PLOTTING ======
PLOT_TRAINING = True
PLOT_ROC = True
PLOT_LR = True
PLOT_SCORE = True
PLOT_SPB = False

#Saving
SAVE_KERAS_MODEL = True

#for the manager script
REPEATS = 3

TAG = f"PN_SR_scan_f=3"
COMMENT = "Replica of R269; with only 3 features"
#This will overwrite the settings above if a grid is used\
#GRID = create_grid("SR={SIGNAL_RATIO:.3f}".format, {"SIGNAL_RATIO": [0.5,0.1,0.05,0.01]})
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
GRID = create_grid(lambda **d: f"SR={d['SN_SLICE']/BG_SLICE*100:.1f}", {"SN_SLICE":[1,500,1000,2000,3000]},
                   N_SN_TRAIN = lambda **d: -1 if d['SN_SLICE']>1 else 0,
                   N_SN_VAL = lambda **d: -1 if d['SN_SLICE']>1 else 0)