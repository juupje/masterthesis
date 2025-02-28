#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from time import time
import h5py
from models import modelhandler
from DataLoaders import Monojets
from models.CustomModels import SupervisedModel
import utils
from utils import shuffling
import re
import argparse
import numpy as np
import callbacks.lr_scheduler as lrs
import logging, logging.config

def setup_logging(loglevel:bool='WARNING', logfile:str=None):
    d = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': loglevel,
            }
        },
        'root': {
            'handlers': ['console', 'file'] if logfile else ["console"],
            'level': 'DEBUG',
        },
    }
    if logfile:
        d['handlers'].update({'file': {
                'class': 'logging.FileHandler',
                'filename': logfile,
                'formatter': 'standard',
                'level': 'DEBUG',
            }})
    logging.config.dictConfig(d)
    logger = logging.getLogger(__name__)
    #log the ID of the current run
    if config:
        if("RUN_ID" in config and config["RUN_ID"] is not None):
            RUN_ID = config["RUN_ID"]
        else:
            RUN_ID = utils.misc.get_run_id(config["RUN_ID_FILE"],inc=False)
        logger.info(f"Using run ID {RUN_ID}")

    #try to log the SLURM Job ID for future reference
    jobid = None
    try:
        with open(os.path.join(outdir, "runtime.txt"), 'w') as f:
            jobid = int(os.environ["SLURM_JOB_ID"])
            f.write(str(jobid)+"\n")
    except:
        logger.warning("Failed to record Job ID")
    return logger

def get_data(config):
    global data
    data = {"data_background":  h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_BG"].replace("?", "0")), mode='r'),
            "data_signal":      h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_SN"].replace("?", "0")), mode='r')}
        
    n_bg_t = int(config["N_DATA_BACKGROUND_TRAIN"])
    n_sn_t = int(config["N_DATA_SIGNAL_TRAIN"])
    if n_bg_t == -1: n_bg_t = data["data_background"]["signal"].shape[0]
    if n_sn_t == -1: n_sn_t = data["data_signal"]["signal"].shape[0]

    if "N_DATA_BACKGROUND_VAL" in config:
        n_bg_v = int(config["N_DATA_BACKGROUND_VAL"])
        n_sn_v = int(config["N_DATA_SIGNAL_VAL"])
    elif "VAL_SPLIT" in config:
        v = config["VAL_SPLIT"]
        logger.debug(f"Using validation split of {v}")
        n_bg_t, n_bg_v = int((1-v)*n_bg_t), int(v*n_bg_t)
        n_sn_t, n_sn_v = int((1-v)*n_sn_t), int(v*n_sn_t)

    data_seed = config.get("DATA_SEED") or config["SEED"]
    logger.info(f"\tData\n\t\tBackground: {n_bg_t:d}/{data['data_background']['signal'].shape[0]}\n\t\tSignal: {n_sn_t:d}/{data['data_signal']['signal'].shape[0]}")
    logger.debug(f"SEED: {data_seed:d}")

    for x in data.keys():
        assert data[x][f"4mom"].shape[1]>=config["N"], f"Number of particles per event does not match config ({x}/4mom)"
        assert data[x][f"features"].shape[1]>=config["N"], f"Number of particles per event does not match config ({x}/features)"
        assert data[x][f"mask"].shape[1]>=config["N"], f"Number of particles per event does not match config ({x}/mask)"
        assert data[x][f"4mom"].shape[2]==4, f"Size of 4-momentum is not 4 ({x}/4mom)"
    if(config["FEATURES"] != 'all'):
        logger.warning("Using features: " + str(config["FEATURES"]))
    if("NOISE_FEATURE" in config):
        logger.warning(f"Adding {config['NOISE_FEATURE']} noise features with params {config['NOISE_PARAM']}")
    #we assume that if the training background is correct, the training signal and validation data will also be correct

    batch_size = config["BATCH_SIZE"]
    steps = int((n_bg_t+n_sn_t)/batch_size)
    inputs = ["4mom","features","mask"]
    if config["MODEL"]=="ParticleNet" or config["MODEL"]=="PermuNet":
        inputs[0] = "coords"

    logger.info(f"{steps} training steps and {int((n_bg_v+n_sn_v)/batch_size)} validation steps")
    train_generator = Monojets.MultiDataLoader_SingleJet(config["N_DATASETS"], sn_file_pattern=os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_SN"]),
                                                                               bg_file_pattern=os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_BG"]),
                                        batch_size=batch_size, N_background=n_bg_t, N_signal=n_sn_t, inputs=inputs, features=config["FEATURES"],
                                        particles=config["N"], seed=data_seed,
                                        noise_features=config.get("NOISE_FEATURES", None), noise_param=config.get("NOISE_PARAM", (0,1)),
                                        data_bg_slice=slice(n_bg_t), data_sn_slice=slice(n_sn_t))
    val_generator = Monojets.MultiDataLoader_SingleJet(config["N_DATASETS"], sn_file_pattern=os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_SN"]),
                                                                               bg_file_pattern=os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_BG"]),
                                        batch_size=batch_size, N_background=n_bg_v, N_signal=n_sn_v, inputs=inputs, features=config["FEATURES"],
                                        particles=config["N"], seed=data_seed,
                                        noise_features=config.get("NOISE_FEATURES", None), noise_param=config.get("NOISE_PARAM", (0,1)),
                                        data_bg_slice=slice(n_bg_t, n_bg_t+n_bg_v), data_sn_slice=slice(n_sn_t, n_sn_t+n_sn_v))
    logger.debug(f"Train generator: {len(train_generator):d}\nVal generator: {len(val_generator):d}")
    for key in data:
        data[key].close()
    return train_generator, val_generator

def get_test_data():
    data_bg = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_BG_TEST"]), mode='r')
    data_sn = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_SN_TEST"]), mode='r')
    #pick the required number of events
    n_test_bg = config["N_TEST_BACKGROUND"] if config["N_TEST_BACKGROUND"]>0 else np.infty
    n_test_sn = config["N_TEST_SIGNAL"] if config["N_TEST_SIGNAL"]>0 else np.infty
    n_bg, n_sn = min(data_bg["mask"].shape[0], n_test_bg), min(data_sn["mask"].shape[0], n_test_sn)
    logger.info(f"Using {n_bg} background and {n_sn} signal test events")
    
    features =  config["FEATURES"]
    data = {}
    for data_name, dataset, n in zip(("background", "signal"), (data_bg, data_sn), (n_bg,n_sn)):
        if(config["MODEL"]=="ParticleNet" or config["MODEL"]=="PermuNet"):
            data[data_name] = {"coords": np.array(dataset["coords"][:n, :config["N"]])}
        else:
            data[data_name] = {"coords": np.array(dataset["4mom"][:n, :config["N"]])}
        data[data_name]["mask"] = np.array(dataset["mask"][:n,:config["N"]])
        if(features):
            data[data_name]["features"] = np.array(dataset["features"][:n, :config["N"]])
            if(features != "all"):
                data[data_name]["features"] = data[data_name]["features"][:,:,features]
    if(config.get("NOISE_FEATURES", None)):
        rng2 = np.random.default_rng(config.get("DATA_SEED") or config["SEED"])
        for name in data.keys():
            data[name]["features"] = np.concatenate((data[name]["features"], rng2.normal(*config.get("NOISE_PARAM", (0,1)), size=(*data[name]["features"].shape[:-1], 1))),axis=-1)
    
    l = ["coords", "features", "mask"] if features else ["coords", "mask"]
    data = [np.concatenate((data["background"][x], data["signal"][x]), axis=0) for x in l]
    labels = np.concatenate((np.zeros(n_bg, dtype=np.int8), np.ones(n_sn, dtype=np.int8)))
    rng = np.random.default_rng(config.get("DATA_SEED") or config["SEED"])
    idx = np.arange(n_bg+n_sn)
    rng.shuffle(idx)
    shuffling.do_the_shuffle(data, idx)
    shuffling.do_the_shuffle(labels, idx)
    return data, labels, idx

def get_callbacks():
    from callbacks.SaveModelCallback import SaveModelCallback
    checkpoint_name = "model-checkpoint_best.keras"
    config["CHECKPOINT_DIR"] = os.path.join(outdir, config["CHECKPOINT_DIR"])
    if not os.path.isdir(config["CHECKPOINT_DIR"]):
        os.makedirs(config["CHECKPOINT_DIR"])
    filepath = os.path.join(config["CHECKPOINT_DIR"], checkpoint_name)

    # Prepare callbacks for model saving
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                monitor=config["MONITOR"],verbose=1, save_best_only=True)
    #progress_bar = tf.keras.callbacks.ProgbarLogger()
    callbacks = [checkpoint]
    if(config["ES_PATIENCE"] is not None):
        from callbacks.earlystopping import EarlyStopping
        callbacks.append(EarlyStopping(min_epoch=config["ES_MIN_EPOCHS"], monitor=config["MONITOR"], patience=config["ES_PATIENCE"]))
    
    if(config["CHECKPOINT_FREQ"] is not None):
        # Prepare callback for model saving
        callbacks.append(SaveModelCallback(directory=config["CHECKPOINT_DIR"],
                            name_fmt="checkpoint-{epoch:d}.keras", freq=config["CHECKPOINT_FREQ"]))
    # just save the model in case something crashes!
    callbacks.append(SaveModelCallback(directory=config["CHECKPOINT_DIR"], name_fmt="checkpoint-latest", freq=1, save_weights=True))
    return callbacks

def get_lr_callback():
    if(config["LR_USE_CHAIN"]):
        return lrs.ChainedScheduler(config["LR_CHAIN"], batch_update=config["LR_PER_BATCH"], steps_per_epoch=len(train_generator), verbose=True)
    else:
        return keras.callbacks.ReduceLROnPlateau(monitor=config["MONITOR"], factor=0.1, patience=config["LR_PATIENCE"])

### -----  ENTRY POINT ------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model trainer")
    parser.add_argument("--config", "-c", help="Path of the configuration file (.json or .py)", required=True, type=str)
    parser.add_argument("--plot_script", "-p", help="Path of the plot script (.py)", type=str)
    parser.add_argument("--outdir", "-o", help="Relative path of the output directory", type=str)
    parser.add_argument("--resume", help="Resume training from this epoch", type=int)
    parser.add_argument("--test", "-t", help="Only perform testing (assumes that the model has been trained already)", action='store_true')
    parser.add_argument("--logfile", "-f", help="Log to file", type=str)
    parser.add_argument("--loglevel", "-l", help="Log level", type=str, choices=["ERROR", "WARNING", "INFO", "DEBUG"], default="INFO")
    parser.add_argument("--no-strategy", "-s", help="Don't use a distributed strategy", action='store_true')
    args = parser.parse_args()
    #print("Using arguments: ", args)
    args = vars(args)

    outdir = os.path.dirname(__file__)
    if(args["outdir"]):
        outdir = os.path.join(outdir, args["outdir"]) #go into the run directory
        if(not os.path.exists(outdir)):
            os.makedirs(outdir)

    ### ----- PREPARATION -----
    config, plotter = None,None
    config = utils.configs.parse_config(args["config"])
    #setup logging
    logger = setup_logging(args['loglevel'], args["logfile"])
    logger.info(f"Using output dir {outdir:s}")
    
    import keras
    import tensorflow as tf
    visible_devices = tf.config.list_physical_devices("GPU")
    assert len(visible_devices)>=1, "visible devices "+str(visible_devices)
    logger.info("Using visible devices: "+str(visible_devices))

    logger.info(f"Using config file {args['config']:s}")

    if(args["plot_script"]):
        if(os.path.exists(args["plot_script"])):
            plotter = utils.imports.import_mod("plotter", args["plot_script"])
        else:
            raise ValueError("Plotting script " + args["plot_script"] + " does not exist.")
        logger.info("Using plot script " + args["plot_script"])

    script_basename = re.sub("(\.py|\.m\.py)$", "", os.path.basename(__file__))

    tic = time()

    ### ----- SETUP -----
    #set the seed
    keras.utils.set_random_seed(config["SEED"])

    #check which strategy we're going to use
    strategy = config.get("STRATEGY", "one_device") if not args["no_strategy"] else "one_device"
    if(strategy == "mirror"):
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    elif(strategy == "one_device"):
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    assert len(visible_devices)==strategy.num_replicas_in_sync, "Not all devices are used by the strategy!"
    logger.info(f"Using strategy {strategy}")
    if(not args["test"]):
        with strategy.scope():
            if(args["resume"]):
                checkpoint = os.path.join(outdir, config["CHECKPOINT_DIR"], "checkpoint-latest.keras")
                if(os.path.exists(checkpoint)):
                    logger.info("!!!!! USING LATEST CHECKPOINT !!!!!")
                else:
                    checkpoint = os.path.join(outdir, config["CHECKPOINT_DIR"], "model-final.keras")
                checkpoint_copy = os.path.join(outdir, config["CHECKPOINT_DIR"], "model_pre-resume.keras")
                os.system(f"cp -r '{checkpoint}' '{checkpoint_copy}'")
                os.mkdir(os.path.join(outdir, "pre-resume"))
                os.system(f"cp {outdir}/*.* {outdir}/pre-resume/")
                model = modelhandler.load_model(checkpoint, config["MODEL"])
                initial_epoch = int(args["resume"])
            else:
                model = modelhandler.create_single_jet_model(config, model_class=SupervisedModel)
                initial_epoch = 0
            #keras.utils.plot_model(model, to_file=os.path.join(outdir, "model.png"), show_layer_names=True)
            train_generator, val_generator = get_data(config)
            callbacks = get_callbacks()
            lr = get_lr_callback()
            toc1 = time()

            ### ----- TRAINING ----
            result = model.fit(train_generator,
                epochs=config["EPOCHS"],
                validation_data=val_generator,
                verbose=config["VERBOSITY"],
                callbacks=callbacks+[lr],
                initial_epoch=initial_epoch)
            model.save(os.path.join(config["CHECKPOINT_DIR"], "model-final.keras"))
            #now, we can delete the `latest` savefile:
            checkpoint = os.path.join(config["CHECKPOINT_DIR"], "checkpoint-latest.keras")
            if(os.path.exists(checkpoint)):
                os.system(f"rm -r \"{checkpoint}\"")
            
            # Store the training history
            file = h5py.File(os.path.join(outdir, "training_stats.h5"), mode='w')
            for key in result.history:
                file.create_dataset(key, data=result.history[key])
            #find the LR scheduler
            if(isinstance(lr, lrs.Scheduler)):
                file.create_dataset("learning_rate", data=lr.history)
            try:
                file.create_dataset("n_signal_per_batch", data=train_generator.get_signals_per_batch())
            except Exception:
                logger.warning("Could not save n_signal_per_batch")
                pass
            file.close()
            #do some clean-up
            train_generator.stop()
            val_generator.stop()
            toc2 = time()
    else:
        logger.debug("Running in test mode!")

    ### ----- PREDICTING/PLOTTING -----
    predictions = {}
    if(config["PERFORM_TEST"]):
        final_model = modelhandler.load_model(os.path.join(outdir, "model-checkpoints","model-final.keras"), config["MODEL"]) if args["test"] else model
        best_model_file = os.path.join(outdir, "model-checkpoints","model-checkpoint_best.keras")
        best_model = modelhandler.load_model(best_model_file, config["MODEL"]) if os.path.isfile(best_model_file) else None
        
        data, labels, idx = get_test_data()
        for model, name in zip([final_model, best_model], ["final", f"best ({config['MONITOR']})"]):
            if(model is None): continue #to catch non-existent best model
            pred = model.predict(data, batch_size=config["BATCH_SIZE"], verbose=config["VERBOSITY"])
            one_hot = np.stack((1-labels, labels), axis=1)
            logger.debug("Test loss:", np.mean(tf.keras.metrics.categorical_crossentropy(one_hot, pred)))
            path = os.path.join(outdir, f"pred_{name.split(' ')[0]}.h5")
            file = h5py.File(path, 'w')
            file.create_dataset("pred", data=pred)
            file.create_dataset("label", data=labels)
            file.create_dataset("idx", data=idx)
            file.close()
            predictions[name] = path
    toc3 = time()
    if(plotter is not None):
        try:
            if(args["test"]):
                extra_info = dict(jobid=os.environ.get("SLURM_JOB_ID",None), train_time=0, test_time=toc3-tic)
            else:
                extra_info = dict(jobid=os.environ.get("SLURM_JOB_ID",None), train_time=toc2-toc1, test_time=toc3-toc2)
            plotter.plot(outdir=outdir, model_name=config["MODELNAME"],
                        training_data_file=os.path.join(outdir, "training_stats.h5") if config["PLOT_TRAINING"] else None,
                        prediction_files=predictions if config["PLOT_ROC"] else None,
                        config=config, extra_info=extra_info, plot_lr=config["PLOT_LR"], plot_spb=config["PLOT_SPB"], plot_score_hist=config["PLOT_SCORE"])
        except Exception as e:
            logger.error(e)
    
    toc4 = time()

    ### ----- FINALIZING -----
    s = ""
    if not args["test"]:
        s = f"Init time: {toc1-tic:.2f}\n" +\
            f"Train time: {toc2-toc1:.2f}\n" +\
            f"Predict time: {toc3-toc2:.2f}\n" +\
            f"Plot time: {toc4-toc3:.2f}\n" +\
            f"Preprocessing:\n\tTraining:{train_generator.get_time_used():.2f}\n\tValidation:{val_generator.get_time_used():.2f}\n" +\
            f"Shuffling:\n\tTraining:{train_generator.get_time_used_shuffle():.2f}\n\tValidation:{val_generator.get_time_used_shuffle():.2f}\n"
        with open(os.path.join(outdir, "runtime.txt"), 'a') as f:
            f.write(s)
    else:
        s = f"Predict time: {toc3-tic:.2f}\n" +\
            f"Plot time: {toc4-toc3:.2f}\n"
        with open(os.path.join(outdir, "runtime-test.txt"), 'w') as f:
            f.write(s)
        logger.info(s)

else:
    config = None
    logger = setup_logging()