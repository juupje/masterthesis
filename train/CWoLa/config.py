#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "CWoLa2"
MODELNAME = "LorentzNet" # used in plots
RUN_ID = None

SLURM = {
    "ACCOUNT": "rwth0934",
    "PARTITION": "c23g",
    "LOGDIR": "auto",#os.path.join(HOME, "out", JOBNAME),
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
SN_SLICE = (0,2_000)

CROSS_VALIDATION = {
    "K": 5,
    "train": [0,1],
    "val": [2,3],
    "test": [4]
}

DO_SR_TEST = True
N_TEST = 100_000
BG_SLICE_TEST = (None,) #since the background data used for testing is in a separate file
SN_SLICE_TEST = (2000, None) #make sure not to pick the test events
# Define the bands
SB_WIDTH=400
SR       = (3300, 3700)
#SR       = (3700, 4100)
SB_LEFT  = (SR[0]-SB_WIDTH, SR[0])
SB_RIGHT = (SR[1], SR[1]+SB_WIDTH)

OVERSAMPLING = 'repeat_strict'

# LR scheduler
MONITOR = 'loss'
LR_PER_BATCH = True

#Data files
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

TAG = f"LNplus_SR_scan"
COMMENT = "The CWoLa setup with LorentzNet+ and a scan over the signal ratio"
#This will overwrite the settings above if a grid is used\
GRID = create_grid(lambda **d: f"SR={d['SN_SLICE'][1]/BG_SLICE[1]*100:.2f}", {"SN_SLICE":[(0,0), (0,200), (0,300), (0,400), (0,600), (0,800), (0,1000), (0,1500), (0,2000), (0,3000)]},
                   SN_SLICE_TEST=lambda **d: (3000,None))
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
#GRID = create_grid("sb_size={SB_WIDTH:d}".format, {"SB_WIDTH":[200,400]},
#                   SB_LEFT = lambda **d: (SR[0]-d["SB_WIDTH"],SR[0]),
#                   SB_RIGHT = lambda **d: (SR[1], SR[1]+d['SB_WIDTH']))
