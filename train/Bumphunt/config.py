#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "SlidingWindow"
MODELNAME = "LorentzNet" # used in plots
RUN_ID = None

SLURM = {
    "ACCOUNT": "rwth0934",
    "PARTITION": "c23g",
    "LOGDIR": os.path.join(HOME, "out", JOBNAME),
    "MEMORY": "10G",
    # Request the time you need for execution. The full format is D-HH:MM:SS
    # You must at least specify minutes OR days and hours and may add or leave out any other parameters      
    "TIME": "60",
    "CONDA_ENV": "tf2"
}

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
EXTRA_DECODER_INPUTS = None #["N_PARTICLES"]

EXTRA_CONFIGS = []
if(MODEL=="PELICAN"):
    EXTRA_CONFIGS.append("config_pelican.py")
elif(MODEL=="LorentzNet" or MODEL=="LorentzNetV2"):
    EXTRA_CONFIGS.append("config_lorentznet.py")
elif(MODEL=="ParticleNet"):
    EXTRA_CONFIGS.append("config_particlenet.py")

# ===== TRAINING =====
CHANGE_SEED = False #for batch processing
SEED = 65
DATA_SEED = 4
BATCH_SIZE = 512
UPDATE_STEPS = 1
VERBOSITY = 2
PERFORM_TEST = True

#Background sizes
FACTOR=1
BG_SLICE = (0,1_000_000)
SN_SLICE = (0,1)

CROSS_VALIDATION = {
    "K": 5,
    "train": [0,1],
    "val": [2,3],
    "test": [4]
}

N_TEST = 100_000
BG_SLICE_TEST = (None,) #since the background data used for testing is in a separate file
SN_SLICE_TEST = (2000, None) #make sure not to pick the test events
# Define the bands
SB_WIDTH=400
SR_WIDTH=400
SR       = (3300, 3700)
SB_LEFT  = (SR[0]-SB_WIDTH, SR[0])
SB_RIGHT = (SR[1], SR[1]+SB_WIDTH)

OVERSAMPLING = 'repeat_strict'

# LR scheduler
MONITOR = 'loss'
LR_PER_BATCH = True

#Data files
#PREPROCESSED_TRAIN_DATA_BG = f"new/extended/N100-bg-combined.h5"
#PREPROCESSED_TRAIN_DATA_SN = f"new/N100-sn.h5"
#PREPROCESSED_TEST_DATA_BG_SR  = f"new/SR_only/N100-bg-SR-small.h5"
#PREPROCESSED_TEST_DATA_SN  = f"new/N100-sn-test.h5"

PREPROCESSED_BG_DATA = "original/N100-bg.h5"
PREPROCESSED_SN_DATA = "original/N100-sn.h5"
PREPROCESSED_TEST_DATA_BG_SR = "original/SR_only/N100-bg_SR-extra.h5"
PREPROCESSED_TEST_DATA_SN_SR = "original/SR_only/N100-sn_SR.h5"

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

TAG = f"LNplus_SWS_SR=0.0_wide-sb"
COMMENT = "Sliding window search with LN+, using 0 signal events and 1M background events and 200GeV sidebands."
SR_CENTERS = [3500+100*i for i in range(-4,5)]
SR_WINDOWS = [(c-SR_WIDTH/2, c+SR_WIDTH/2) for c in SR_CENTERS]
SB_WINDOWS = [[(w[0]-SB_WIDTH, w[0]), (w[1], w[1]+SB_WIDTH)] for w in SR_WINDOWS]

#This will overwrite the settings above if a grid is used
#          window 0     window 1      window 2      window 3      window 4      window 5      window 6      window 7      window 8      window 9
#left = [None,       (2103, 2300), (2288, 2500), (2461, 2700), (2632, 2900), (2788, 3100), (2911, 3300), (3111, 3500), (3311, 3700), (3511, 3900)]
#sr   = [None,       (2300, 2700), (2500, 2900), (2700, 3100), (2900, 3300), (3100, 3500), (3300, 3700), (3500, 3900), (3700, 4100), (3900, 4300)]
#right= [None,       (2700, 2897), (2900, 3112), (3100, 3339), (3300, 3568), (3816, 3766), (3700, 4090), (3900, 4290), (4100, 4490), (4300, 4690)]

GRID = create_grid(lambda **d: f"window={d['WINDOW']}", {"WINDOW":list(range(len(SR_CENTERS)))},
                   SB_LEFT = lambda **d: SB_WINDOWS[d['WINDOW']][0],
                   SR = lambda **d: SR_WINDOWS[d['WINDOW']],
                   SB_RIGHT = lambda **d: SB_WINDOWS[d['WINDOW']][1])
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")