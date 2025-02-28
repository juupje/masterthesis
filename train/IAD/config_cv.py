#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "IAD_crossval"
MODELNAME = "LorentzNet" # used in plots
RUN_ID = None

SLURM = {
    "ACCOUNT": "rwth0934",
    "PARTITION": "c23g",
    "LOGDIR": os.path.join(HOME, "out", JOBNAME),
    "MEMORY": "10G",
    # Request the time you need for execution. The full format is D-HH:MM:SS
    # You must at least specify minutes OR days and hours and may add or leave out any other parameters      
    "TIME": "100",
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
#DATA_DIR = os.path.join(os.getenv("DATA_DIR"), "lhco")
DATA_DIR = "/hpcwork/rwth0934/LHCO_dataset/processed_jg/recreated"
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
TEST_SR_VS_BG = True

#Background sizes
FACTOR = 1
N_SIMULATED = 272_000*FACTOR # taken from sim-bg
N_DATA_BACKGROUND = 120_000*FACTOR # taken from data-bg
N_DATA_SIGNAL = 790*FACTOR # taken from data-sn

CROSS_VALIDATION = {
    "K": 5,
    "train": [0,1],
    "val": [2,3],
    "test": [4]
}

N_TEST_BACKGROUND  = 200_000 # taken from sim-bg
N_TEST_SIGNAL = 22_000 # taken from data-sn

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
MONITOR = "loss"
LR_PER_BATCH = True

#Data files
PREPROCESSED_DATA_BG_SIM  = "N100-sim-bg-SR_unrotated.h5"
PREPROCESSED_DATA_BG_DATA = "N100-data-bg-SR_unrotated.h5"
PREPROCESSED_DATA_SN_DATA = "N100-data-sn-SR_unrotated.h5"

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

TAG = "kfold+repeat"
COMMENT = 'Testing k-fold cross-validation including 5 repeats'
#This will overwrite the settings above if a grid is used\
#GRID = create_grid(lambda DATA_SEED: f"{DATA_SEED=:d}", {"DATA_SEED": [0,1,2,3,4]})
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
#GRID = create_grid(lambda N_DATA_SIGNAL_TRAIN: f"S={N_DATA_SIGNAL_TRAIN:d}", {"N_DATA_SIGNAL_TRAIN": [x*FACTOR for x in [200, 300, 395, 600,800,1000, 1500,2000]]},
#                   N_DATA_SIGNAL_VAL = lambda **d: int(d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO),
#                   N_SIMULATED_VAL = lambda **d: int(N_DATA_BACKGROUND_VAL+d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO))
