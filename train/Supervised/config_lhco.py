#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "Supervised"
MODELNAME = "LorentzNetV2" # used in plots
RUN_ID = None

SLURM = {
    "ACCOUNT": "rwth0934",
    "PARTITION": "c23g",
    "LOGDIR": os.path.join(HOME, "out", JOBNAME),
    "MEMORY": "24G",
    # Request the time you need for execution. The full format is D-HH:MM:SS
    # You must at least specify minutes OR days and hours and may add or leave out any other parameters      
    "TIME": "3:00:00",
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
DATA_DIR = "/hpcwork/kd106458/data"
OUTPUT_DIR = f"{WORK_DIR}/outputs/{JOBNAME}"
RUN_ID_FILE = f"{WORK_DIR}/run_id.json"
CHECKPOINT_DIR = "model-checkpoints"
CHECKPOINT_FREQ = None

# ===== MODEL ======
MODEL = MODELNAME #used to load the model

N=200

EXTRA_CONFIGS = []
if(MODEL=="PELICAN"):
    EXTRA_CONFIGS.append("config_pelican.py")
elif(MODEL.startswith("LorentzNet")):
    EXTRA_CONFIGS.append("config_lorentznet.py")
elif(MODEL=="ParticleNet"):
    EXTRA_CONFIGS.append("config_particlenet.py")

# ===== TRAINING =====
CHANGE_SEED = True #for batch processing
SEED = 123
DATA_SEED = 10
# BATCH_SIZE = 500 set by model config
VERBOSITY = 2
PERFORM_TEST = True

#Background sizes
N_DATA_BACKGROUND_TRAIN = -1 # taken from data-bg
N_DATA_SIGNAL_TRAIN = -1 # taken from data-sn

# now, we want to keep the relative frequencies of events equal to training
VAL_SPLIT = 0.1

N_TEST_BACKGROUND  = -1 # taken from sim-bg
N_TEST_SIGNAL = -1 # taken from data-sn

OVERSAMPLING = None

# Define the bands
#SB_LEFT  = (None, 3300)
#SR        = (3300, 3700)
#SB_RIGHT = (3700, None)

# LR scheduler
'''LR_START = 2.5e-3
LR_USE_CHAIN = False
LR_PATIENCE = 3
ES_PATIENCE = 6'''
MONITOR = "val_loss"
LR_PER_BATCH = True

#Data files
directory="lhco/mono"
PREPROCESSED_DATA_BG = f"{directory}/mono_bg_train_SR.h5"
PREPROCESSED_DATA_SN = f"{directory}/mono_sn_train_SR.h5"
PREPROCESSED_DATA_BG_TEST = f"{directory}/mono_bg_test_SR.h5"
PREPROCESSED_DATA_SN_TEST = f"{directory}/mono_sn_test_SR.h5"
#PREPROCESSED_DATA_BG  = f"toptagging/train_bg.h5"
#PREPROCESSED_DATA_SN = f"toptagging/train_sn.h5"
#PREPROCESSED_DATA_BG_TEST = f"toptagging/test_bg.h5"
#PREPROCESSED_DATA_SN_TEST = f"toptagging/test_sn.h5"
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

TAG = "LN_mono"
COMMENT = f"small-data test of {MODELNAME} performance on LHCO data in monojet format"
#This will overwrite the settings above if a grid is used\
#GRID = create_grid(lambda LR_MAX, LR_MIN: f"LR={LR_MAX:.0e}->-{LR_MIN:.0e}", {"LR_MAX": [5e-3, 1e-2], "LR_MIN": [1e-5, 1e-4]})
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
#GRID = create_grid(lambda N_DATA_SIGNAL_TRAIN: f"S={N_DATA_SIGNAL_TRAIN:d}", {"N_DATA_SIGNAL_TRAIN": [395, 600*factor, 1200*factor]},
#                   N_DATA_SIGNAL_VAL = lambda **d: int(d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO),
#                   N_SIMULATED_VAL = lambda **d: int(N_DATA_BACKGROUND_VAL+d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO))
#GRID = create_grid(lambda FEATURES: f"f={len(FEATURES):d}", {"FEATURES": [[2], [2,6]]})
GRID = create_grid("N={N:d}".format, {"N": [1, 2, 10, 40, 128]})