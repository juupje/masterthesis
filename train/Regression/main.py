#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from time import time
import h5py
from models import modelhandler
from DataLoaders import RegressionDataLoader as RDL
import utils
import re
import argparse
import numpy as np
import callbacks.lr_scheduler as lrs
import json
import logging, logging.config

def setup_logging(console_debug:bool=False, log_file:str=None):
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
                'level': 'DEBUG' if console_debug else "INFO",
            }
        },
        'root': {
            'handlers': ['console', 'file'] if log_file else ["console"],
            'level': 'DEBUG',
        },
    }
    if log_file:
        d['handlers'].update({'file': {
                'class': 'logging.FileHandler',
                'filename': log_file,
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
    data = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_TRAIN"]))
    # For training
    n_train = int(config["N_TRAIN"])
    if n_train == -1: n_train = data["signal"].shape[0]
    # For validation
    if "N_VAL" in config:
       n_val = config["N_VAL"]
    else:
        v = config["VAL_SPLIT"]
        n_train, n_val = int((1-v)*n_train), int(v*n_train)
    data_seed = config.get("DATA_SEED") or config["SEED"]
    logger.info(f"\tData:\n\t\ttraining {n_train:d}/{data['signal'].shape[0]}\n\t\tvalidation: {n_val:d}/{data['signal'].shape[0]}")
    logger.debug(f"SEED: {data_seed:d}")

    for jet in ["jet1", "jet2"]:
        assert data[f"{jet}/coords"].shape[1]>config["N"], f"Number of particles per event does not match config ({jet})"
        assert data[f"{jet}/4mom"].shape[2]==4, f"Size of 4-momentum is not 4 ({jet})"
        assert data[f"{jet}/coords"].shape[2]==2, f"Number of coordinates is not 4 ({jet})"
        assert data[f"{jet}/mask"].shape[2]==1, f"Mask depth is not 1 ({jet})"
    if(config["FEATURES"] != 'all'):
        logger.warning("Using features: " + str(config["FEATURES"]))
    if("NOISE_FEATURE" in config):
        logger.warning(f"Adding {config['NOISE_FEATURE']} noise features with params {config['NOISE_PARAM']}")
    #we assume that if the training background is correct, the training signal and validation data will also be correct
    batch_size = config["BATCH_SIZE"]
    steps = int(n_train/batch_size)
    inputs = ["4mom","features","mask"]
    if config["MODEL"]=="ParticleNet":
        inputs[0] = "coords"

    logger.info(f"{steps} training steps and {int(n_val/batch_size)} validation steps")
    
    train_slice = slice(0,n_train)
    val_slice = slice(n_train, n_train+n_val)

    if config.get("MERGE_JETS", False):
        DL = RDL.MergedRegressionDataLoader
    else:
        DL = RDL.RegressionDataLoader
    train_generator = DL(data, batch_size, data_slice=train_slice, regression_feature=-1, particles=config["N"],
                                               log=True, shift_mean=True,
                                               N_data=n_train, features=config["FEATURES"], seed=config["SEED"], inputs=inputs, njets=2)
    val_generator = DL(data, batch_size, data_slice=val_slice, regression_feature=-1, particles=config["N"],
                                            log=True, shift_mean=train_generator.get_shift(),
                                            N_data=n_val, features=config["FEATURES"], seed=config["SEED"], inputs=inputs, njets=2)
    print(f"Train generator: {len(train_generator):d}\nVal generator: {len(val_generator):d}")
    data.close()
    return train_generator, val_generator

def get_test_data():
    if "PREPROCESSED_DATA_TEST" not in config:
        file = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_TRAIN"]))
        offset = config["N_TRAIN"] + config.get("N_VAL", 0)
        logger.warning(f"Reusing training dataset for testing, with offset {offset}")
    else:
        file = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_TEST"]), mode='r')
        offset = 0
    l = ["coords", "features", "mask"] if config["MODEL"] == "ParticleNet" else ["4mom", "features", "mask"]
    jets = ["jet1", "jet2"]
    features = config["FEATURES"]
    if(features is None):
        l.remove("features")
    #only select events in the Signal Region (SR)
    #pick the required number of events
    n_test = config["N_TEST"]
    if n_test+offset > file["signal"].shape[0]:
        logger.error("Not enough test data, using what is available")
        n_test = file["signal"].shape[0]-offset
    logger.info(f"Using {n_test} test events")
    data = {jet: {x:np.array(file[jet][x][offset:offset+n_test,:config["N"]]) for x in l} for jet in jets}
    if(features != 'all' and features is not None):
        #select the required features
        for jet in jets:
            data[jet]["features"] = data[jet]["features"][:,:,features]
    target = np.array(file["jet_features"][offset:offset+n_test,-1])
    file.close()
    if "MERGE_JETS" in config:
        newdata = {}
        for s in l:
            newdata[s] = np.concatenate([data[jet][s] for jet in jets], axis=1)
        del data
        return [newdata[s] for s in l], target, np.log10(target), np.arange(n_test)
    else:
        return [data[jet][s] for jet in jets for s in l], target, np.log10(target), np.arange(n_test)

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
    if(config["LR_USE_CHAIN"]):
        chain = lrs.ChainedScheduler(config["LR_CHAIN"], batch_update=config["LR_PER_BATCH"], steps_per_epoch=len(train_generator), verbose=True)
        callbacks.append(chain)
    else:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor=config["MONITOR"], factor=0.1, patience=config["LR_PATIENCE"]))
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

### -----  ENTRY POINT ------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model trainer")
    parser.add_argument("--config", "-c", help="Path of the configuration file (.json or .py)", required=True, type=str)
    parser.add_argument("--plot_script", "-p", help="Path of the plot script (.py)", type=str)
    parser.add_argument("--outdir", "-o", help="Relative path of the output directory", type=str)
    parser.add_argument("--resume", help="Resume training from this epoch", type=int)
    parser.add_argument("--test", "-t", help="Only perform testing (assumes that the model has been trained already)", action='store_true')
    parser.add_argument("--debug", "-d", help="Enabled debug logging", nargs='?', default=False, const=True)
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
    if args['debug'] == False:
        logger = setup_logging(False)
    else:
        logger = setup_logging(True, os.path.join(outdir, args['debug']) if type(args['debug']) is str else None)
    
    import keras
    import tensorflow as tf
    visible_devices = tf.config.list_physical_devices("GPU")
    assert len(visible_devices)==1, "visible devices "+str(visible_devices)
    logger.info("Using visible devices: "+str(visible_devices))

    logger.info(f"Using config file {args['config']:s}")

    if(args["plot_script"]):
        if(os.path.exists(args["plot_script"])):
            plotter = utils.imports.import_mod("plotter", args["plot_script"])
        else:
            raise ValueError("Plotting script " + args["plot_script"] + " does not exist.")
        logger.info("Using plot script " + args["plot_script"])

    script_basename = re.sub("(\.py|\.m\.py)$", "", os.path.basename(__file__))
    logger.info(f"Using output dir {outdir:s}")

    tic = time()

    ### ----- SETUP -----
    #set the seed
    keras.utils.set_random_seed(config["SEED"])
    if(not args["test"]):
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
            modelconfig = config
            if("MERGE_JETS" in modelconfig):
                modelconfig = modelconfig.copy()
                modelconfig["N"] = 2*modelconfig["N"]

            model = modelhandler.create_regression_model(modelconfig)
            initial_epoch = 0
        train_generator, val_generator = get_data(config)
        callbacks = get_callbacks()
        toc1 = time()

        #save the shift transformation
        with open("trafo.json", 'w') as f:
            json.dump({"log":True, "shift":float(train_generator.get_shift())}, f)

        ### ----- TRAINING ----
        result = model.fit(train_generator,
            epochs=config["EPOCHS"],
            validation_data=val_generator,
            verbose=config["VERBOSITY"],
            callbacks=callbacks,
            initial_epoch=initial_epoch)
        model.save(os.path.join(config["CHECKPOINT_DIR"], "model-final.keras")) #h5 file
        #now, we can delete the `latest` savefile:
        checkpoint = os.path.join(config["CHECKPOINT_DIR"], "checkpoint-latest.keras")
        if(os.path.exists(checkpoint)):
            os.system(f"rm -r \"{checkpoint}\"")
        
        # Store the training history
        file = h5py.File(os.path.join(outdir, "training_stats.h5"), mode='w')
        for key in result.history:
            file.create_dataset(key, data=result.history[key])
        #find the LR scheduler
        for callback in callbacks:
            if(isinstance(callback, lrs.Scheduler)):
                file.create_dataset("learning_rate", data=callback.history)
                break
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
        
        data, target, log_target, idx = get_test_data()
        with open("trafo.json", 'r') as f:
            trafo:dict = json.load(f)
        assert trafo.get("log"), "No log transformation was used"
        for model, name in zip([final_model, best_model], ["final", f"best ({config['MONITOR']})"]):
            if(model is None): continue #to catch non-existent best model
            pred = model.predict(data, batch_size=config["BATCH_SIZE"], verbose=config["VERBOSITY"])
            pred = pred+trafo.get("shift", 0)
            print("Test loss (log):", np.mean(np.square(np.squeeze(pred)-np.squeeze(log_target))))
            reco = np.power(10, pred)
            print("Test loss:", np.mean(np.square(np.squeeze(reco)-np.squeeze(target))))
            path = os.path.join(outdir, f"pred_{name.split(' ')[0]}.h5")
            file = h5py.File(path, 'w')
            file.create_dataset("log_reco", data=np.squeeze(pred,axis=-1))
            file.create_dataset("reco", data=np.squeeze(reco,axis=-1))
            file.create_dataset("target", data=target)
            file.create_dataset("data_idx", data=idx)
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
                        prediction_files=predictions if config["PLOT_RECO"] else None,
                        config=config, extra_info=extra_info, plot_lr=config["PLOT_LR"])
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