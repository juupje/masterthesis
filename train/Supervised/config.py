#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "Supervised"
MODELNAME = "LorentzNet" # used in plots
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
EPOCHS = 35
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
BATCH_SIZE = 128
VERBOSITY=1
PERFORM_TEST = True

#Background sizes
N_BACKGROUND_TRAIN = 136_000 # SR background from extra simulation
N_SIGNAL_TRAIN = 27_000 # SR signal from LHCO
N_BACKGROUND_VAL = 136_000 # SR background from extra simulation
N_SIGNAL_VAL = 27_000  # SR signal from LHCO
N_TEST_BACKGROUND  = 340_000 # SR background from extra simulation
N_TEST_SIGNAL = 20_000 # SR signal from LHCO
VAL_SPLIT=0.5 #unused

BG_SLICE = N_BACKGROUND_TRAIN
SN_SLICE = N_SIGNAL_TRAIN
BG_SLICE_VAL = N_BACKGROUND_VAL
SN_SLICE_VAL = N_SIGNAL_VAL

NOISE_TYPE = 'normal'

# Define the bands
#SB_LEFT  = (None, 3300)
SR       = (3300, 3700)
#SB_RIGHT = (3700, None)

# LR scheduler
'''
LR_USE_CHAIN = True
LR_START = 2.5e-3
LR_PATIENCE = 3
ES_PATIENCE = 6
'''
LR_PER_BATCH=True
MONITOR = "val_loss"
#Data files
PREPROCESSED_BG_DATA = f"new/N30-sim-bg-big.h5"
PREPROCESSED_BG_DATA = f"new/N30-data-sn-big.h5"
PREPROCESSED_TEST_DATA_BG  = f"new/N30-sim-bg-small.h5"
PREPROCESSED_TEST_DATA_SN  = f"new/N30-data-sn-small.h5"

# ====== PLOTTING ======
PLOT_TRAINING = True
PLOT_ROC = True
PLOT_LR = True
PLOT_SCORE = True

#Saving
SAVE_KERAS_MODEL = True

#for the manager script
REPEATS = 5

TAG = f"cathode_extra_decoder"
#This will overwrite the settings above if a grid is used\
#GRID = create_grid("c={C:.1e}".format, {"C": [1e-6,5e-5, 1e-5, 5e-4,1e-4,5e-3]})
GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
                    MODELNAME = "{MODEL:s}".format,
                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
