#!/usr/bin/env python
# coding: utf-8

#A supervised PELICAN tensorflow implementation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from time import time
import h5py
from models import modelhandler
from DataLoaders import CWoLaDataLoader as CWL, Discriminator
import jlogger as jl
import utils
import re
import argparse
import numpy as np
import lr_scheduler as lrs

def setup_logging():
    # --------- LOGGING -------------- 
    #log the ID of the current run
    if("RUN_ID" in config and config["RUN_ID"] is not None):
        RUN_ID = config["RUN_ID"]
    else:
        RUN_ID = utils.get_run_id(config["RUN_ID_FILE"],inc=False)
    print(f"Using run ID {RUN_ID}")

    #try to log the SLURM Job ID for future reference
    jobid = None
    try:
        with open(os.path.join(outdir, "runtime.txt"), 'w') as f:
            jobid = int(os.environ["SLURM_JOB_ID"])
            f.write(str(jobid)+"\n")
    except:
        print("Failed to record Job ID")

    #create the logger
    return jl.JLogger(jobid=jobid, output_dir=outdir, comment=f"R{RUN_ID:d}")

def get_data(test:bool=False) -> tuple[CWL.CWoLaDataLoader]|CWL.CWoLaDataLoader:
    global data
    data = {"background": h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_TRAIN_DATA_BG"])),
                "signal": h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_TRAIN_DATA_SN"]))}
    # For training
    n_bg_t = config["N_BG_TRAIN"]
    n_sn_t = config["N_SN_TRAIN"]
    assert n_sn_t>=0 or n_sn_t==-1, "Invalid number of signal events (training)"
    assert n_bg_t>0 or n_bg_t==-1, "Invalid number of background data (training)"

    # For validation
    n_bg_v = config["N_BG_VAL"]
    n_sn_v = config["N_SN_VAL"]
    assert n_sn_v>=0 or n_sn_v==-1, "Invalid number of signal events (validation)"
    assert n_bg_v>0 or n_bg_v==-1, "Invalid number of background events (validation)"

    data_seed = config.get("DATA_SEED") or config["SEED"]
    print(f"Data:\n\tBackground: {n_bg_t:d}/{data['background']['signal'].shape[0]}\n\tSignal: {n_sn_t:d}/{data['signal']['signal'].shape[0]}")
    print(f"SEED: {data_seed:d}")

    for jet in ["jet1", "jet2"]:
        for x in data.keys():
            assert data[x][f"{jet}/coords"].shape[1]>config["N"], f"Number of particles per event does not match config ({x}: {jet})"
            assert data[x][f"{jet}/4mom"].shape[2]==4, f"Size of 4-momentum is not 4 ({x}: {jet})"
            assert data[x][f"{jet}/coords"].shape[2]==2, f"Number of coordinates is not 4 ({x}: {jet})"
            assert data[x][f"{jet}/mask"].shape[2]==1, f"Mask depth is not 1 ({x}: {jet})"
    if(config["FEATURES"] != 'all'):
        print("WARNING: using features: ", config["FEATURES"])
    if("NOISE_FEATURE" in config):
        print(f"WARNING: adding {config['NOISE_FEATURE']} noise features with params {config['NOISE_PARAM']}")
    #we assume that if the training background is correct, the training signal and validation data will also be correct

    batch_size = config["BATCH_SIZE"]
    inputs = ["coords", "features", "mask"] if config["MODEL"] == "ParticleNet" else ["4mom", "features", "mask"]
    
    bg_slice, sn_slice = config["BG_SLICE"], config["SN_SLICE"]

    if(not(test)):
        train_generator = CWL.CWoLaDataLoader(data["background"], data["signal"], batch_size, particles=config["N"],
                                    SB_left=config["SB_LEFT"], SR=config["SR"], SB_right=config["SB_RIGHT"],
                                    background_slice=slice(0,bg_slice), signal_slice=slice(0,sn_slice),oversampling=config["OVERSAMPLING"],
                                    N_background=n_bg_t, N_signal=n_sn_t, features=config["FEATURES"], seed=data_seed, inputs=inputs, njets=2, do_shuffle=True)
        print(f"Train generator: {len(train_generator):d}")
        val_generator = CWL.CWoLaDataLoader(data["background"], data["signal"], batch_size, particles=config["N"],
                                    SB_left=config["SB_LEFT"], SR=config["SR"], SB_right=config["SB_RIGHT"],
                                    background_slice=slice(bg_slice,bg_slice*2), signal_slice=slice(sn_slice, sn_slice*2),
                                    N_background=n_bg_v, N_signal=n_sn_v, features=config["FEATURES"], seed=data_seed, inputs=inputs, njets=2,
                                    oversampling=config["OVERSAMPLING"], include_true_labels=True, do_shuffle=True)
        print(f"Val generator: {len(val_generator):d}")
        for key in data: data[key].close() #some cleanup
        return train_generator, val_generator
    else:
        test_generator = CWL.CWoLaDataLoader(data["background"], data["signal"], batch_size,
                                    SB_left=config["SB_LEFT"], SR=config["SR"], SB_right=config["SB_RIGHT"],
                                    background_slice=slice(bg_slice*2, bg_slice*3), signal_slice=slice(sn_slice*2, sn_slice*3),oversampling=False,
                                    N_background=n_bg_t, N_signal=n_sn_t, features=config["FEATURES"], seed=data_seed, inputs=inputs, njets=2, do_shuffle=False)
        print(f"Test generator: {len(test_generator):d}")
        for key in data: data[key].close() #some cleanup
        return test_generator

def get_SR_test_data():
    data_bg = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_TEST_DATA_BG"]), mode='r')
    data_sn = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_TEST_DATA_SN"]), mode='r')
    l = ["coords", "features", "mask"] if config["MODEL"] == "ParticleNet" else ["4mom", "features", "mask"]
    jets = ["jet1", "jet2"]
    features = config["FEATURES"]
    if(features is None):
        l.remove("features")
    disc = Discriminator.Discriminator("jet_features/-1", lower=config["SR"][0], upper=config["SR"][1])
    #only select events in the Signal Region (SR)
    is_sr_background = disc.apply(data_bg)
    is_sr_signal = disc.apply(data_sn)
    #count how many of those events we have
    idx_bg = np.where(is_sr_background)[0]
    idx_sn = np.where(is_sr_signal)[0]
    #pick the required number of events
    n_test = config["N_TEST"] if config["N_TEST"]>0 else np.infty
    n_bg, n_sn = min(idx_bg.shape[0], n_test), min(idx_sn.shape[0], n_test)
    print(f"Using {n_bg} background and {n_sn} signal test events")
    idx_bg = idx_bg[:n_bg]#np.random.choice(idx_bg, n_bg, replace=False)
    idx_sn = idx_sn[:n_sn]#np.random.choice(idx_sn, n_sn, replace=False)
    idx = np.concatenate([idx_bg, -idx_sn-1], axis=0)
    
    bg = {jet: {x:np.array(data_bg[jet][x][:,:config["N"]])[idx_bg] for x in l} for jet in jets}
    sn = {jet: {x:np.array(data_sn[jet][x][:,:config["N"]])[idx_sn] for x in l} for jet in jets}
    if(features != 'all' and features is not None):
        #select the required features
        for jet in jets:
            bg[jet]["features"] = bg[jet]["features"][:,:,features]
            sn[jet]["features"] = sn[jet]["features"][:,:,features]
    data = [np.concatenate((bg[jet][x], sn[jet][x]),axis=0) for jet in jets for x in l]
    labels = np.concatenate((np.zeros(n_bg, dtype=np.int8), np.ones(n_sn, dtype=np.int8)),axis=0)
    return data, labels, idx

def get_test_data():
    file = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_TRAIN_DATA_BG"]), mode='r')
    l = ["coords", "features", "mask"] if config["MODEL"] == "ParticleNet" else ["4mom", "features", "mask"]
    jets = ["jet1", "jet2"]
    features = config["FEATURES"]
    if(features is None):
        l.remove("features")
    disc_left = Discriminator.Discriminator("jet_features/-1", lower=config["SB_LEFT"][0], upper=config["SB_LEFT"][1])
    disc_middle = Discriminator.Discriminator("jet_features/-1", lower=config["SR"][0], upper=config["SR"][1])
    disc_right = Discriminator.Discriminator("jet_features/-1", lower=config["SB_RIGHT"][0], upper=config["SB_RIGHT"][1])
    test_slice = slice(config["BG_SLICE"]*2, None)
    #only select events in the Signal Region (SR)
    is_sb_or_sr = np.logical_or(np.logical_or(
                    disc_left.apply(file, test_slice),
                    disc_right.apply(file, test_slice)),
                    disc_middle.apply(file, test_slice))
    #count how many of those events we have
    idx = np.where(is_sb_or_sr)[0]
    #pick the required number of events
    n_test = config["BG_SLICE"]
    n = min(idx.shape[0], n_test)
    print(f"Using {n} signal region and side band test events")
    idx = idx[:n]
    
    data = {jet: {x:np.array(file[jet][x][test_slice,:config["N"]])[idx] for x in l} for jet in jets}
    if(features != 'all' and features is not None):
        #select the required features
        for jet in jets:
            data[jet]["features"] = data[jet]["features"][:,:,features]
    mjj = np.array(file["jet_features"][test_slice,-1])[idx]
    labels = np.logical_and(mjj>=config["SR"][0], mjj<=config["SR"][1])
    print(f"\twith {np.sum(labels):d} SR events")
    file.close()
    return [data[jet][x] for jet in jets for x in l], labels, idx, mjj

def get_callbacks():
    from SaveModelCallback import SaveModelCallback
    checkpoint_name = "model-checkpoint_best.h5"
    config["CHECKPOINT_DIR"] = os.path.join(outdir, config["CHECKPOINT_DIR"])
    if not os.path.isdir(config["CHECKPOINT_DIR"]):
        os.makedirs(config["CHECKPOINT_DIR"])
    filepath = os.path.join(config["CHECKPOINT_DIR"], checkpoint_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                monitor=config["MONITOR"],verbose=1, save_best_only=True)
    #progress_bar = tf.keras.callbacks.ProgbarLogger()
    callbacks = [checkpoint]
    if(config["LR_USE_CHAIN"]):
        chain = lrs.ChainedScheduler(config["LR_CHAIN"], batch_update=config["LR_PER_BATCH"], steps_per_epoch=len(train_generator), verbose=True)
        callbacks.append(chain)
    else:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=config["MONITOR"], factor=0.1, patience=config["LR_PATIENCE"]))
    if(config["ES_PATIENCE"] is not None):
        from earlystopping import EarlyStopping
        callbacks.append(EarlyStopping(min_epoch=config["ES_MIN_EPOCHS"], monitor=config["MONITOR"], patience=config["ES_PATIENCE"]))
    
    if(config["CHECKPOINT_FREQ"] is not None):
        # Prepare callback for model saving
        callbacks.append(SaveModelCallback(directory=config["CHECKPOINT_DIR"],
                            name_fmt="checkpoint-{epoch:d}.h5", freq=config["CHECKPOINT_FREQ"]))
    callbacks.append(SaveModelCallback(directory=config["CHECKPOINT_DIR"], name_fmt="checkpoint-latest", freq=1))
    return callbacks

### -----  ENTRY POINT ------
if __name__ == "__main__":
    import tensorflow as tf
    visible_devices = tf.config.list_physical_devices("GPU")
    assert len(visible_devices)==1, "visible devices "+str(visible_devices)
    print("Using visible devices: ", visible_devices)

    ### ----- PREPARATION -----
    parser = argparse.ArgumentParser("Model trainer")
    parser.add_argument("--config", "-c", help="Path of the configuration file (.json or .py)", required=True, type=str)
    parser.add_argument("--plot_script", "-p", help="Path of the plot script (.py)", type=str)
    parser.add_argument("--outdir", "-o", help="Relative path of the output directory", type=str)
    parser.add_argument("--resume", help="Resume training from this epoch", type=int)
    parser.add_argument("--test", "-t", help="Only perform testing (assumes that the model has been trained already)", action='store_true')
    args = parser.parse_args()
    print("Using arguments: ", args)
    args = vars(args)

    config, plotter = None,None
    config = utils.parse_config(args["config"])
    print(f"Using config file {args['config']:s}")

    if(args["plot_script"]):
        if(os.path.exists(args["plot_script"])):
            plotter = utils.import_mod("plotter", args["plot_script"])
        else:
            raise ValueError("Plotting script " + args["plot_script"] + " does not exist.")
        print("Using plot script " + args["plot_script"])

    script_basename = re.sub("(\.py|\.m\.py)$", "", os.path.basename(__file__))
    outdir = os.path.dirname(__file__)
    if(args["outdir"]):
        outdir = os.path.join(outdir, args["outdir"]) #go into the run directory
    print(f"Using output dir {outdir:s}")

    tic = time()

    ### ----- SETUP -----
    #set the seed
    tf.keras.utils.set_random_seed(config["SEED"])
    #setup logging
    logger = setup_logging()
    if(not args["test"]):
        if(args["resume"]):
            checkpoint = os.path.join(outdir, config["CHECKPOINT_DIR"], "checkpoint-latest")
            if(os.path.exists(checkpoint)):
                print("!!!!! USING LATEST CHECKPOINT !!!!!")
            else:
                checkpoint = os.path.join(outdir, config["CHECKPOINT_DIR"], "model-final" if config["SAVE_KERAS_MODEL"] else "model-final.h5")
            checkpoint_copy = os.path.join(outdir, config["CHECKPOINT_DIR"], "model_pre-resume" if config["SAVE_KERAS_MODEL"] else "model_pre-resume.h5")
            os.system(f"cp -r '{checkpoint}' '{checkpoint_copy}'")
            os.mkdir(os.path.join(outdir, "pre-resume"))
            os.system(f"cp {outdir}/*.* {outdir}/pre-resume/")
            model = modelhandler.load_model(checkpoint, config["MODEL"])
            initial_epoch = int(args["resume"])
        else:
            model = modelhandler.create_dijet_model(config)
            initial_epoch = 0
        train_generator, val_generator = get_data()
        callbacks = get_callbacks()
        toc1 = time()

        ### ----- TRAINING ----
        weights = {0:1, 1:(1 if config["OVERSAMPLING"] else 1/train_generator.get_signal_ratio())}
        print("WEIGHTS:", weights)
        result = model.fit(train_generator,
            class_weight=weights,
            epochs=config["EPOCHS"],
            validation_data=val_generator,
            verbose=config["VERBOSITY"],
            callbacks=callbacks,
            initial_epoch=initial_epoch)
        model.save(os.path.join(config["CHECKPOINT_DIR"], "model-final.h5")) #h5 file
        if(config["SAVE_KERAS_MODEL"]):
            model.save(os.path.join(config["CHECKPOINT_DIR"], "model-final")) #full keras save (for experimental adamw optimizer)
        #now, we can delete the `latest` savefile:
        checkpoint = os.path.join(config["CHECKPOINT_DIR"], "checkpoint-latest")
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
        file.create_dataset("n_signal_per_batch", data=train_generator.get_signals_per_batch())
        file.close()
        #do some clean-up
        for key in data: data[key].close()
        train_generator.stop()
        val_generator.stop()
        toc2 = time()
    else:
        print("Running in test mode!")

    ### ----- PREDICTING/PLOTTING -----
    predictions = {}
    if(config["PERFORM_TEST"]):
        final_model = modelhandler.load_model(os.path.join(outdir, "model-checkpoints","model-final"), config["MODEL"], ) if args["test"] else model
        best_model_file = os.path.join(outdir, "model-checkpoints","model-checkpoint_best.h5")
        best_model = modelhandler.load_model(best_model_file, config["MODEL"]) if os.path.isfile(best_model_file) else None
        
        #First, test signal vs. background
        
        sr_data, sr_labels, sr_idx = get_SR_test_data()
        test_generator = get_data(test=True)
        data = [test_generator.data[jet][s] for jet in test_generator.jets for s in test_generator.inputs]
        cwola_labels = test_generator.labels["cwola"][:,1]
        true_labels = test_generator.labels["true"][:,1]
        is_cr = cwola_labels==0
        print(f"CR size: {np.sum(is_cr):d}")
        print(f"SR size: {np.sum(~is_cr):d}")
        print(data[0].shape)
        for model, name in zip([final_model, best_model], ["final", f"best ({config['MONITOR']})"]):
            if(model is None): continue #to catch non-existent best model
            pred = model.predict(sr_data, batch_size=config["BATCH_SIZE"], verbose=config["VERBOSITY"])
            one_hot = np.stack((1-sr_labels, sr_labels), axis=1)
            print("Test loss:", np.mean(tf.keras.metrics.categorical_crossentropy(one_hot, pred)))
            path = os.path.join(outdir, f"pred_{name.split(' ')[0]}.h5")
            file = h5py.File(path, 'w')
            group = file.create_group("s_vs_b")
            group.create_dataset("pred", data=pred)
            group.create_dataset("label", data=sr_labels)
            group.create_dataset("data_idx", data=sr_idx)
            
            pred = model.predict(data, batch_size=config["BATCH_SIZE"], verbose=config["VERBOSITY"])
            group_cr = file.create_group("cr")
            group_cr.create_dataset("pred", data=pred[is_cr])
            group_cr.create_dataset("label", data=true_labels[is_cr])
            group_sr = file.create_group("sr")
            group_sr.create_dataset("pred", data=pred[~is_cr])
            group_sr.create_dataset("label", data=true_labels[~is_cr])
            file.close()
            predictions[name] = path
    toc3 = time()
    if(plotter is not None):
        try:
            if(args["test"]):
                extra_info = dict(jobid=logger.jobid, train_time=0, test_time=toc3-tic)
            else:
                extra_info = dict(jobid=logger.jobid, train_time=toc2-toc1, test_time=toc3-toc2)
            plotter.plot(outdir=outdir, logger=logger, model_name=config["MODELNAME"],
                        training_data_file=os.path.join(outdir, "training_stats.h5") if config["PLOT_TRAINING"] else None,
                        prediction_files=predictions if config["PLOT_ROC"] else None,
                        config=config, extra_info=extra_info, plot_lr=config["PLOT_LR"], plot_spb=config["PLOT_SPB"], plot_score_hist=config["PLOT_SCORE"])
        except Exception:
            import traceback
            traceback.print_exc()
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
        print(s)
