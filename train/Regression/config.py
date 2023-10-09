#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "regression"
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
BG_SLICE_TRAIN = 500_000
BG_SLICE_VAL = 500_000

N_TRAIN = 360_000

VAL_RATIO = 0.2
N_VAL = int(VAL_RATIO*N_TRAIN)

N_TEST = 100_000
# Define the bands
'''
SB_LEFT  = (2600, 3300)
SR       = (3300, 3700)
SB_RIGHT = (3700, 5000)
'''

# LR scheduler
MONITOR = 'val_loss'
LR_PER_BATCH = True

#Data files
PREPROCESSED_TRAIN_DATA = f"new/N100-bg.h5"
PREPROCESSED_TEST_DATA  = f"new/N100-bg-test.h5"

# ====== PLOTTING ======
PLOT_TRAINING = True
PLOT_RECO = True
PLOT_LR = True

#Saving
SAVE_KERAS_MODEL = True

#for the manager script
REPEATS = 2

TAG = f"test1_shift"
COMMENT = "Included shift"
#This will overwrite the settings above if a grid is used\
#GRID = create_grid("SR={SIGNAL_RATIO:.3f}".format, {"SIGNAL_RATIO": [0.5,0.1,0.05,0.01]})
GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
                    MODELNAME = "{MODEL:s}".format,
                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
