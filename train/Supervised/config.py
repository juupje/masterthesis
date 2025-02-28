#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "Transformer"
MODELNAME = "PELICAN" # used in plots
RUN_ID = None

SLURM = {
    "ACCOUNT": "rwth0934",
    "PARTITION": "c23g",
    "LOGDIR": os.path.join(HOME, "out", JOBNAME),
    "MEMORY": "32G",
    "GPUS": 2,
    # Request the time you need for execution. The full format is D-HH:MM:SS
    # You must at least specify minutes OR days and hours and may add or leave out any other parameters      
    "TIME": "2-00:00:00",
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
#DATA_DIR = "/hpcwork/kd106458/data/transformer"
OUTPUT_DIR = f"{WORK_DIR}/outputs/{JOBNAME}"
RUN_ID_FILE = f"{WORK_DIR}/run_id.json"
CHECKPOINT_DIR = "model-checkpoints"
CHECKPOINT_FREQ = None

# ===== MODEL ======
MODEL = MODELNAME #used to load the model

N=128

EXTRA_CONFIGS = []
if(MODEL=="PermuNet"):
    EXTRA_CONFIGS.append("config_permunet.py")
elif(MODEL=="PELICAN"):
    EXTRA_CONFIGS.append("config_pelican.py")
elif(MODEL.startswith("LorentzNet")):
    EXTRA_CONFIGS.append("config_lorentznet.py")
elif(MODEL=="ParticleNet"):
    EXTRA_CONFIGS.append("config_particlenet.py")

STRATEGY = "mirror"
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
VAL_SPLIT = 0.2

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
directory="transformer"
#PREPROCESSED_DATA_BG  = f"{directory}/qcd_train_nonan.h5"
#PREPROCESSED_DATA_SN = f"{directory}/original_binned/qcd_train_nonan.h5"
#PREPROCESSED_DATA_BG_TEST = f"{directory}/qcd_test_nonan.h5"
#PREPROCESSED_DATA_SN_TEST = f"{directory}/original_binned/qcd_test_nonan.h5"
PREPROCESSED_DATA_BG  = f"{directory}/10M/qcd_?_pre.h5"
PREPROCESSED_DATA_SN = f"{directory}/10M/top_?_pre.h5"
PREPROCESSED_DATA_BG_TEST = f"{directory}/10M/qcd_test_pre.h5"
PREPROCESSED_DATA_SN_TEST = f"{directory}/10M/top_test_pre.h5"
N_DATASETS = 5
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
REPEATS = 5

TAG = "PL_full_shortLR_2gpu"
COMMENT = f"Large-data test of {MODELNAME} performance with shorter LR schedule on two gpus"
#This will overwrite the settings above if a grid is used\
#GRID = create_grid(lambda LR_MAX, LR_MIN: f"LR={LR_MAX:.0e}->-{LR_MIN:.0e}", {"LR_MAX": [5e-3, 1e-2], "LR_MIN": [1e-5, 1e-4]})
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
#GRID = create_grid(lambda N_DATA_SIGNAL_TRAIN: f"S={N_DATA_SIGNAL_TRAIN:d}", {"N_DATA_SIGNAL_TRAIN": [395, 600*factor, 1200*factor]},
#                   N_DATA_SIGNAL_VAL = lambda **d: int(d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO),
#                   N_SIMULATED_VAL = lambda **d: int(N_DATA_BACKGROUND_VAL+d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO))
#GRID = create_grid(lambda FEATURES: f"f={len(FEATURES):d}", {"FEATURES": [[2], [2,6]]})
#GRID = create_grid("N={N:d}".format, {"N": [2, 10, 20, 40, 128]},
#                PREPROCESSED_DATA_BG= lambda **d: f"{directory}/N{d['N']}_qcd_train_nonan.h5",
#                PREPROCESSED_DATA_SN = lambda **d: f"{directory}/N{d['N']}_top_train_nonan.h5",
#                PREPROCESSED_DATA_BG_TEST = lambda **d: f"{directory}/N{d['N']}_qcd_test_nonan.h5",
#                PREPROCESSED_DATA_SN_TEST = lambda **d: f"{directory}/N{d['N']}_top_test_nonan.h5")