#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "Supervised"
MODELNAME = "ParticleNet" # used in plots
RUN_ID = None

SLURM = {
    "ACCOUNT": "rwth0934",
    "PARTITION": "c23g",
    "LOGDIR": os.path.join(HOME, "out", JOBNAME),
    "MEMORY": "12G",
    # Request the time you need for execution. The full format is D-HH:MM:SS
    # You must at least specify minutes OR days and hours and may add or leave out any other parameters      
    "TIME": "00:30:00",
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
EXTRA_CONFIGS = []
if(MODEL=="PELICAN"):
    EXTRA_CONFIGS.append("config_pelican.py")
elif(MODEL.startswith("LorentzNet")):
    EXTRA_CONFIGS.append("config_lorentznet.py")
elif(MODEL=="ParticleNet"):
    EXTRA_CONFIGS.append("config_particlenet.py")

# ===== TRAINING =====
CHANGE_SEED = True #for batch processing
SEED = 4
DATA_SEED=2
VERBOSITY=2
PERFORM_TEST = True

N=40
MERGE_JETS = False
N_DATA_BACKGROUND_TRAIN = -1 # SR background
N_DATA_SIGNAL_TRAIN = -1 # SR signal
VAL_SPLIT = 0.25

N_DATA_BACKGROUND_TEST  = -1 # SR background
N_DATA_SIGNAL_TEST = -1 # SR signal
OVERSAMPLING = None

BG_SLICE = None
SN_SLICE = None
NOISE_TYPE = 'normal'
WEIGHT_CLASSES = True

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
directory = "efp/ordered/singlejet/jet1"
PREPROCESSED_DATA_BG = f"{directory}/bg_train_SR.h5"
PREPROCESSED_DATA_SN = f"{directory}/sn_train_SR.h5"
PREPROCESSED_DATA_BG_TEST  = f"{directory}/bg_test_SR.h5"
PREPROCESSED_DATA_SN_TEST  = f"{directory}/sn_test_SR.h5"

DECODER_INPUTS = ["Mjet1", "Mjet2", "JetSeparation"]

# ====== PLOTTING ======
PLOT_TRAINING = True
PLOT_ROC = True
PLOT_LR = True
PLOT_SCORE = True

#Saving
SAVE_KERAS_MODEL = True

#for the manager script
REPEATS = 2

TAG = f"PN_only_jet1_DF"
COMMENT = f"Testing only single LHCO jets with the {MODELNAME} model"
#This will overwrite the settings above if a grid is used\
#GRID = create_grid("c={C:.1e}".format, {"C": [1e-6,5e-5, 1e-5, 5e-4,1e-4,5e-3]})
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
#GRID = create_grid("Jet{jetidx:d}".format, {"jetidx": [1,2]},
#                    JET_NOISE = lambda **d: {"index": d['jetidx']-1, "type": 'zero'})