#configuration file
import os
HOME = os.getenv("HOME")
JOBNAME = "KerasV3"
MODELNAME = "LorentzNetV2" # used in plots
RUN_ID = None

SLURM = {
    "ACCOUNT": "rwth0934",
    "PARTITION": "c23g",
    "LOGDIR": os.path.join(HOME, "out", JOBNAME),
    "MEMORY": "10G",
    # Request the time you need for execution. The full format is D-HH:MM:SS
    # You must at least specify minutes OR days and hours and may add or leave out any other parameters      
    "TIME": "300",
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
BATCH_SIZE = 128
UPDATE_STEPS = 1
VERBOSITY = 2
PERFORM_TEST = True
TEST_SR_VS_BG = False

#Background sizes
FACTOR = 1
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
MONITOR = "val_loss"
LR_PER_BATCH = True

#Data files
#PREPROCESSED_DATA_BG_SIM  = "ranit/2prong_1000/wrangled_simulated_zero_masked.h5"
#PREPROCESSED_DATA_BG_DATA = "ranit/2prong_1000/wrangled_background_zero_masked.h5"
#PREPROCESSED_DATA_SN_DATA = "ranit/2prong_1000/wrangled_signal_zero_masked.h5"
#PREPROCESSED_DATA_BG_SIM  = "new/SR_only/N30-sim-bg_SR_unrotated.h5"
#PREPROCESSED_DATA_BG_DATA = "new/SR_only/N30-data-bg_SR_unrotated.h5"
#PREPROCESSED_DATA_SN_DATA = "new/SR_only/N30-data-sn_SR_unrotated.h5"

PREPROCESSED_DATA_BG_SIM  = "N100-sim-bg-SR_unrotated.h5"
PREPROCESSED_DATA_BG_DATA = "N100-data-bg-SR_unrotated.h5"
PREPROCESSED_DATA_SN_DATA = "N100-data-sn-SR_unrotated.h5"

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

TAG = "Keras3_test"
COMMENT = 'Testing the new keras 3 implementation'
#This will overwrite the settings above if a grid is used
'''
GRID = create_grid(lambda N,MODEL: f"{N=}" + ("" if MODEL=="LorentzNet" else "_edgemask"),
                    {"MODEL": ["LorentzNet", "LorentzNetV2"], "N": [50, 70, 90]},
                    BATCH_SIZE = lambda **d: 128 if d["N"]==50 else (64 if d["N"]==70 else 32),
                    UPDATE_STEPS = lambda **d: 4 if d["N"]==50 else (8 if d["N"]==70 else 16),
                    MODELNAME = "{MODEL:s}".format,
                    EXTRA_CONFIGS = lambda **d: f"config_lorentznet.py")
GRID = create_grid(lambda N_DATA_SIGNAL_TRAIN: f"S={N_DATA_SIGNAL_TRAIN:d}", {"N_DATA_SIGNAL_TRAIN": [400*FACTOR, 1000*FACTOR, 2000*FACTOR, 4000*FACTOR]},
                   N_DATA_SIGNAL_VAL = lambda **d: int(d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO),
                   N_SIMULATED_VAL = lambda **d: int(N_DATA_BACKGROUND_VAL+d["N_DATA_SIGNAL_TRAIN"]*VAL_RATIO))
'''
GRID = create_grid(lambda MODEL: f"{MODEL:s}",
                    {"MODEL": ["ParticleNet", "PELICAN"]},
                    MODELNAME = "{MODEL:s}".format,
                    EXTRA_CONFIGS = lambda **d: f"config_{d['MODEL'].lower()}.py")
