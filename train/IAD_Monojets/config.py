#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "IAD_monojets_LN"
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
DATA_DIR = "/hpcwork/rwth0934/cwola_project"
OUTPUT_DIR = f"{WORK_DIR}/outputs/{JOBNAME}"
RUN_ID_FILE = f"{WORK_DIR}/run_id.json"
CHECKPOINT_DIR = "model-checkpoints"
CHECKPOINT_FREQ = None

# ===== MODEL ======
MODEL = MODELNAME #used to load the model
OPTIMIZER = "Adam"
WEIGHT_DECAY = 0.01
N=40

EXTRA_CONFIGS = []
if(MODEL=="PELICAN"):
    EXTRA_CONFIGS.append("config_pelican.py")
elif(MODEL=="LorentzNet"):
    EXTRA_CONFIGS.append("config_lorentznet.py")
elif(MODEL=="ParticleNet"):
    EXTRA_CONFIGS.append("config_particlenet.py")

# ===== TRAINING =====
CHANGE_SEED = True #for batch processing
SEED = 12
DATA_SEED = 1
# BATCH_SIZE = 500 set by model config
VERBOSITY = 2
PERFORM_TEST = True

#Background sizes
FACTOR = 1
N_SIMULATED_TRAIN = 1_000_000*FACTOR # taken from sim-bg
N_DATA_BACKGROUND_TRAIN = 990_000*FACTOR # taken from data-bg
N_DATA_SIGNAL_TRAIN = 10_000*FACTOR # taken from data-sn

# now, we want to keep the number of data events equal to training, and choose
# simulated events such that no weighting is needed
VAL_RATIO = 0.1
N_DATA_BACKGROUND_VAL = "$<|N_DATA_BACKGROUND_TRAIN*VAL_RATIO|>" # taken from data-bg
N_DATA_SIGNAL_VAL = "$<|int(N_DATA_SIGNAL_TRAIN*VAL_RATIO)|>" # taken from data-sn
N_SIMULATED_VAL = "$<|N_DATA_BACKGROUND_VAL+N_DATA_SIGNAL_VAL|>" # taken from sim-bg

N_TEST_BACKGROUND  = 100_000 # taken from sim-bg
N_TEST_SIGNAL = 100_000 # taken from data-sn

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
MONITOR = "loss"
LR_PER_BATCH = True

#Data files
PREPROCESSED_DATA_BG  = f"data_zvv_train.h5" #we use 'big' and 'small' because they're smaller datasets
PREPROCESSED_DATA_SN = f"data_mzp2000_train.h5"
PREPROCESSED_DATA_BG_TEST = f"data_zvv_test.h5"
PREPROCESSED_DATA_SN_TEST = f"data_mzp2000_test.h5"
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

TAG = "less_data"
COMMENT = 'Replica of R251; Less data'
#This will overwrite the settings above if a grid is used\
#GRID = create_grid(lambda LR_MAX, LR_MIN: f"LR={LR_MAX:.0e}->-{LR_MIN:.0e}", {"LR_MAX": [5e-3, 1e-2], "LR_MIN": [1e-5, 1e-4]})
#GRID = create_grid("{MODEL:s}".format, {"MODEL": ["PELICAN", "LorentzNet", "ParticleNet"]},
#                    MODELNAME = "{MODEL:s}".format,
#                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower():s}.py")
GRID = create_grid(lambda FACTOR: f"F={FACTOR:.2f}", {"FACTOR": [0.25, 0.5]},
                    N_SIMULATED_TRAIN = lambda **d: int(N_SIMULATED_TRAIN*d["FACTOR"]), # taken from sim-bg
                    N_DATA_BACKGROUND_TRAIN = lambda **d: int(N_DATA_BACKGROUND_TRAIN*d['FACTOR']), # taken from data-bg
                    N_DATA_SIGNAL_TRAIN = lambda **d: int(10_000*d['FACTOR']) # taken from data-sn
                    )
