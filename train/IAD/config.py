#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "IAD3_cathode_LN"
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
CHANGE_SEED = False #for batch processing
SEED = 65
DATA_SEED = 4
# BATCH_SIZE = 500 set by model config
VERBOSITY = 2
PERFORM_TEST = True

#Background sizes
FACTOR = 1
N_SIMULATED_TRAIN = 136_000*FACTOR # taken from sim-bg
N_DATA_BACKGROUND_TRAIN = 60_000*FACTOR # taken from data-bg
N_DATA_SIGNAL_TRAIN = 395*FACTOR # taken from data-sn

# now, we want to keep the number of data events equal to training, and choose
# simulated events such that no weighting is needed
N_DATA_BACKGROUND_VAL = N_DATA_BACKGROUND_TRAIN//FACTOR # taken from data-bg
VAL_RATIO = N_DATA_BACKGROUND_VAL/N_DATA_BACKGROUND_TRAIN #unused
N_DATA_SIGNAL_VAL = int(N_DATA_SIGNAL_TRAIN*VAL_RATIO) #taken from data-sn
N_SIMULATED_VAL = N_DATA_BACKGROUND_VAL+N_DATA_SIGNAL_VAL # taken from sim-bg

N_TEST_BACKGROUND  = 200_000 # taken from sim-bg
N_TEST_SIGNAL = 200_000 # taken from data-sn

OVERSAMPLING =  'repeat_strict'

# Define the bands
#SB_LEFT  = (None, 3300)
SR        = (3300, 3700)
#SB_RIGHT = (3700, None)

# LR scheduler
'''LR_START = 2.5e-3
LR_USE_CHAIN = False
LR_PATIENCE = 3
ES_PATIENCE = 6'''
MONITOR = "val_loss"
LR_PER_BATCH = True

#Data files
PREPROCESSED_DATA_BG_SIM  = f"new/N$<|N|>-sim-bg-big.h5" #we use 'big' and 'small' because they're smaller datasets
PREPROCESSED_DATA_BG_DATA = f"new/N$<|N|>-data-bg.h5"
PREPROCESSED_DATA_SN_DATA = f"new/N$<|N|>-data-sn-small.h5"
REUSE_BACKGROUND = False

# ====== PLOTTING ======
PLOT_TRAINING = True
PLOT_ROC = True
PLOT_LR = True
PLOT_SCORE = False
PLOT_SPB = False

#Saving
SAVE_KERAS_MODEL = True

#for the manager script
REPEATS = 5

TAG = "base_case"
COMMENT = 'Replica of 188; just to be sure :)'
#This will overwrite the settings above if a grid is used\
#GRID = create_grid(lambda DATA_SEED: f"{DATA_SEED=:d}", {"DATA_SEED": [0,1,2,3,4]})
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
GRID = create_grid(lambda N_DATA_SIGNAL_TRAIN: f"S={N_DATA_SIGNAL_TRAIN:d}", {"N_DATA_SIGNAL_TRAIN": [150*FACTOR, 400*FACTOR]},
                   N_DATA_SIGNAL_VAL = lambda **d: int(d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO),
                   N_SIMULATED_VAL = lambda **d: int(N_DATA_BACKGROUND_VAL+d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO))
