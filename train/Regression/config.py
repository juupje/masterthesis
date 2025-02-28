#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "regression"
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
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.01
N=30
MERGE_JETS = True

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
DATA_SEED = 2
BATCH_SIZE = 256
VERBOSITY=2
PERFORM_TEST = True

#Background sizes
N_TRAIN = 360000 
VAL_SPLIT = 0.1

N_TEST  = 200000

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
directory="lhco/original"
PREPROCESSED_DATA_TRAIN = f"{directory}/N100-bg.h5"
#PREPROCESSED_DATA_TEST =  f"{directory}/sim-bg_unrotated_N100.h5"

# ====== PLOTTING ======
PLOT_TRAINING = True
PLOT_RECO = True
PLOT_LR = True

#Saving
SAVE_KERAS_MODEL = True

#for the manager script
REPEATS = 3

TAG = f"test1_shift"
COMMENT = "Included shift"
#This will overwrite the settings above if a grid is used\
#GRID = create_grid("SR={SIGNAL_RATIO:.3f}".format, {"SIGNAL_RATIO": [0.5,0.1,0.05,0.01]})
GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
                    MODELNAME = "{MODEL:s}".format,
                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
