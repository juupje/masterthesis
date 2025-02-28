#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "IAD2_cathode_PN"
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
SEED = 12365
DATA_SEED = None
# BATCH_SIZE = 802 set by model config file
VERBOSITY = 2
PERFORM_TEST = True

#Background sizes
FACTOR = 2
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
PLOT_SCORE = True
PLOT_SPB = False

#Saving
SAVE_KERAS_MODEL = True

#for the manager script
REPEATS = 3

TAG = "double_data"
COMMENT = 'Replica of R264; double data'
#This will overwrite the settings above if a grid is used\
#GRID = create_grid(lambda N_DATA_SIGNAL_TRAIN,LR_START: f"S={N_DATA_SIGNAL_TRAIN/1000:.1f}k_LR={LR_START:.0e}".replace(".0",""), {"N_DATA_SIGNAL_TRAIN": [395*factor,1000*factor, 2000*factor], "LR_START": [3e-4, 1e-4, 5e-5, 1e-5]},
#                   N_DATA_SIGNAL_VAL = lambda **d: int(d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO),
#                   N_SIMULATED_VAL = lambda **d: int(N_DATA_BACKGROUND_VAL+d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO),
#                   LR_FINAL = lambda **d: d["LR_START"]*10)
#GRID = create_grid(lambda LR_MAX, LR_MIN: f"LR={LR_MAX:.0e}->-{LR_MIN:.0e}", {"LR_MAX": [5e-3, 1e-2], "LR_MIN": [1e-5, 1e-4]})
GRID = create_grid(lambda N_DATA_SIGNAL_TRAIN: f"S={N_DATA_SIGNAL_TRAIN:d}", {"N_DATA_SIGNAL_TRAIN": [2000*FACTOR, 800*FACTOR, 600*FACTOR, 395*FACTOR]},
                   N_DATA_SIGNAL_VAL = lambda **d: int(d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO),
                   N_SIMULATED_VAL = lambda **d: int(N_DATA_BACKGROUND_VAL+d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO))
