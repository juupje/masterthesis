#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "CWoLa"
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
FACTOR = 2
BG_SLICE = 560000
SN_SLICE = 1000
N_BG_TRAIN = -1
N_SN_TRAIN = -1

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
PREPROCESSED_TRAIN_DATA_BG = f"new/extended/N100-bg-combined.h5"
PREPROCESSED_TRAIN_DATA_SN = f"new/N100-sn.h5"
PREPROCESSED_TEST_DATA_BG_SR  = f"new/SR_only/N100-bg-SR-small.h5"
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
REPEATS = 5

TAG = f"LN_sliding_window_SR=0.1_double"
COMMENT = "Replica of R276, but with double background data (halved SR)"
#This will overwrite the settings above if a grid is used
#          window 0     window 1      window 2      window 3      window 4      window 5      window 6      window 7      window 8      window 9
left = [(2017, 2213), (2216, 2414), (2415, 2614), (2588, 2811), (2727, 2987), (2834, 3150), (2911, 3300), (3111, 3500), (3311, 3700), (3511, 3900)]
sr   = [(2213, 2387), (2414, 2586), (2614, 2786), (2811, 2989), (2987, 3213), (3150, 3450), (3300, 3700), (3500, 3900), (3700, 4100), (3900, 4300)]
right= [(2387, 2583), (2586, 2783), (2786, 2985), (2989, 3212), (3213, 3472), (3450, 3766), (3700, 4090), (3900, 4290), (4100, 4490), (4300, 4690)]

GRID = create_grid(lambda **d: f"window={d['WINDOW']}", {"WINDOW":list(range(10))},
                   SB_LEFT = lambda **d: left[d['WINDOW']],
                   SR = lambda **d: sr[d['WINDOW']],
                   SB_RIGHT = lambda **d: right[d['WINDOW']])
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
