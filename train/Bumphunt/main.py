#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, "/home/kd106458/thesis_devel/lib")
from time import time
import h5py
from DataLoaders import CWoLaDataLoader as CWL, Discriminator
import jlogger as jl
import utils
import re
import argparse
import numpy as np
from models import modelhandler
import callbacks.lr_scheduler as lrs

def setup_logging():
    # --------- LOGGING -------------- 
    #log the ID of the current run
    if("RUN_ID" in config and config["RUN_ID"] is not None):
        RUN_ID = config["RUN_ID"]
    else:
        RUN_ID = utils.get_run_id(config["RUN_ID_FILE"],inc=False)
    print(f"Using run ID {RUN_ID}")

    #try to log the SLURM Job ID for future reference
    jobid = os.getenv("SLURM_JOB_ID") or ""
    arrayjob = os.getenv("SLURM_ARRAY_JOB_ID") or ""
    arrayid = os.getenv("SLURM_ARRAY_TASK_ID") or ""
    try:
        with open(os.path.join(outdir, "runtime.txt"), 'w') as f:
            f.write("JobID: "+str(jobid)+"\n")
            f.write("Array Job ID: "+str(arrayjob)+"\n")
            f.write("Array Task ID: "+str(arrayid)+"\n")
    except:
        print("Failed to record Job ID")

    #create the logger
    return {"jobid": jobid, "arrayjob": arrayjob, "arrayid": arrayid}

data = None
def open_data():
    global data
    if data is None:
        data = {"background":  h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_BG_DATA"])),
                "signal":      h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_SN_DATA"]))}
def close_data():
    global data
    if data is not None:
        for key in data: data[key].close()
        data=None

def validate_data():
    assert data is not None, "Data not initialized"
    for jet in ["jet1", "jet2"]:
        for x in data.keys():
            if(config["MODEL"]=="ParticleNet"):
                assert data[x][f"{jet}/coords"].shape[2]==2, f"Number of coordinates is not 2 ({x}: {jet})"
                assert data[x][f"{jet}/coords"].shape[1]>=config["N"], f"Number of particles per event does not match config ({x}: {jet})"
            else:
                assert data[x][f"{jet}/4mom"].shape[2]==4, f"Size of 4-momentum is not 4 ({x}: {jet})"
                assert data[x][f"{jet}/4mom"].shape[1]>=config["N"], f"Number of particles per event does not match config ({x}: {jet})"
            assert data[x][f"{jet}/mask"].shape[2]==1, f"Mask depth is not 1 ({x}: {jet})"
    if(config["FEATURES"] != 'all'):
        print("WARNING: using features: ", config["FEATURES"])
    if("NOISE_FEATURE" in config):
        print(f"WARNING: adding {config['NOISE_FEATURE']} noise features with params {config['NOISE_PARAM']}")

def get_decoder_inputs():
    if config.get("EXTRA_DECODER_INPUTS",None) is not None:
        from DataLoaders import DecoderFeatures as DF
        return [DF.get(name) for name in config["EXTRA_DECODER_INPUTS"]]
    else:
        return None

def get_kfold_data(fold_nr, splits:list[str]):
    open_data()
    validate_data()
    bg_slice=slice(*config["BG_SLICE"])
    sn_slice=slice(*config["SN_SLICE"])
    print(f"Background: {bg_slice.stop-bg_slice.start:d}/{data['background']['signal'].shape[0]}\nSignal: {sn_slice.stop-sn_slice.start:d}/{data['signal']['signal'].shape[0]}")
    data_seed = config.get("DATA_SEED") or config["SEED"]
    print(f"SEED: {data_seed:d}")

    inputs = ["coords", "features", "mask"] if config["MODEL"] == "ParticleNet" else ["4mom", "features", "mask"]

    cv_params = config["CROSS_VALIDATION"].copy()
    cv_params["index"] = fold_nr
    #take the correct slides according to the split
    gens = []
    for split in splits:
        cv_params["split"] = config["CROSS_VALIDATION"][split]
        gens.append(gen := CWL.CWoLaDataLoader(data["background"], data["signal"], batch_size=config["BATCH_SIZE"],
                        background_slice=bg_slice, signal_slice=sn_slice, SB_left=config["SB_LEFT"], SR=config["SR"], SB_right=config["SB_RIGHT"],
                        SR_feature_idx=-1, features=config["FEATURES"], seed=data_seed, inputs=inputs,
                        njets=2, particles=config["N"], do_shuffle=True, oversampling=(config["OVERSAMPLING"] if split!='test' else None),
                        noise_features=config.get("NOISE_FEATURES", None), noise_param=config.get("NOISE_PARAM", None),
                        cross_validation_params=cv_params,
                        include_true_labels=(split!='train')))
        print(f"{split} generator: using {len(gen)} steps")
    return gens

def get_data(test:bool=False) -> tuple[CWL.CWoLaDataLoader]|CWL.CWoLaDataLoader:
    open_data()
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
    validate_data()

    batch_size = config["BATCH_SIZE"]
    inputs = ["coords", "features", "mask"] if config["MODEL"] == "ParticleNet" else ["4mom", "features", "mask"]
    
    bg_slice, sn_slice = config["BG_SLICE"], config["SN_SLICE"]
    bg_slice_t = bg_slice*config["FACTOR"]
    if(not(test)):
        train_generator = CWL.CWoLaDataLoader(data["background"], data["signal"], batch_size, particles=config["N"],
                                    SB_left=config["SB_LEFT"], SR=config["SR"], SB_right=config["SB_RIGHT"],
                                    background_slice=slice(0,bg_slice_t), signal_slice=slice(0,sn_slice),oversampling=config["OVERSAMPLING"],
                                    N_background=n_bg_t, N_signal=n_sn_t, features=config["FEATURES"], seed=data_seed, inputs=inputs, njets=2, do_shuffle=True)
        print(f"Train generator: {len(train_generator):d}")
        val_generator = CWL.CWoLaDataLoader(data["background"], data["signal"], batch_size, particles=config["N"],
                                    SB_left=config["SB_LEFT"], SR=config["SR"], SB_right=config["SB_RIGHT"],
                                    background_slice=slice(bg_slice_t,bg_slice_t+bg_slice), signal_slice=slice(sn_slice, sn_slice*2),
                                    N_background=n_bg_v, N_signal=n_sn_v, features=config["FEATURES"], seed=data_seed, inputs=inputs, njets=2,
                                    oversampling=config["OVERSAMPLING"], include_true_labels=True, do_shuffle=True)
        print(f"Val generator: {len(val_generator):d}")
        for key in data: data[key].close() #some cleanup
        return train_generator, val_generator
    else:
        test_generator = CWL.CWoLaDataLoader(data["background"], data["signal"], batch_size,
                                    SB_left=config["SB_LEFT"], SR=config["SR"], SB_right=config["SB_RIGHT"],
                                    background_slice=slice(bg_slice_t, bg_slice_t+bg_slice), signal_slice=slice(sn_slice*2, sn_slice*3),oversampling=False,
                                    N_background=n_bg_t, N_signal=n_sn_t, features=config["FEATURES"], seed=data_seed, inputs=inputs, njets=2, do_shuffle=False)
        print(f"Test generator: {len(test_generator):d}")
        for key in data: data[key].close() #some cleanup
        return test_generator

def get_callbacks():
    from callbacks.SaveModelCallback import SaveModelCallback
    checkpoint_name = "model-checkpoint_best.weights.h5"
    config["CHECKPOINT_DIR"] = os.path.join(outdir, config["CHECKPOINT_DIR"])
    if not os.path.isdir(config["CHECKPOINT_DIR"]):
        os.makedirs(config["CHECKPOINT_DIR"])
    filepath = os.path.join(config["CHECKPOINT_DIR"], checkpoint_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                monitor=config["MONITOR"],verbose=1, save_weights_only=True, save_best_only=True)

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
                            name_fmt="checkpoint-{epoch:d}", freq=config["CHECKPOINT_FREQ"]))
    if(config.get("CHECKPOINT_FREQ", 0) != 1):
        callbacks.append(SaveModelCallback(directory=config["CHECKPOINT_DIR"], name_fmt="checkpoint-latest", freq=1))
    return callbacks

### -----  ENTRY POINT ------
if __name__ == "__main__":
    import keras
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
    parser.add_argument("--fold", "-f", help="Specify the cross validation fold", type=int)
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
    
    if("CROSS_VALIDATION" in config):
        assert args["fold"] is not None, "No cross validation fold specified"
        cross_validation_fold = args["fold"]
        outdir = os.path.join(outdir, f"fold{cross_validation_fold}")
        os.makedirs(outdir, exist_ok=True)
    else:
        cross_validation_fold = None
    print(f"Using output dir {outdir:s}")

    tic = time()

    ### ----- SETUP -----
    #set the seed
    keras.utils.set_random_seed(config["SEED"])
    #setup logging
    loginfo = setup_logging()
    if(not args["test"]):
        if(args["resume"]):
            checkpoint = os.path.join(outdir, config["CHECKPOINT_DIR"], "checkpoint-latest.keras")
            if(os.path.exists(checkpoint)):
                print("!!!!! USING LATEST CHECKPOINT !!!!!")
            else:
                checkpoint = os.path.join(outdir, config["CHECKPOINT_DIR"], "model-final.keras")
            checkpoint_copy = os.path.join(outdir, config["CHECKPOINT_DIR"], "model_pre-resume.keras")
            os.system(f"cp -r '{checkpoint}' '{checkpoint_copy}'")
            os.mkdir(os.path.join(outdir, "pre-resume"))
            os.system(f"cp {outdir}/*.* {outdir}/pre-resume/")
            model = modelhandler.load_model(checkpoint, config["MODEL"])
            initial_epoch = int(args["resume"])
        else:
            model = modelhandler.create_dijet_model(config)
            initial_epoch = 0
        if cross_validation_fold is not None:
            train_generator, val_generator = get_kfold_data(cross_validation_fold, ["train","val"])
        else:
            train_generator, val_generator = get_data()
        callbacks = get_callbacks()
        toc1 = time()

        ### ----- TRAINING ----
        result = model.fit(train_generator,
            epochs=config["EPOCHS"],
            validation_data=val_generator,
            verbose=config["VERBOSITY"],
            callbacks=callbacks,
            initial_epoch=initial_epoch)
        model.save(os.path.join(config["CHECKPOINT_DIR"], "model-final.keras")) #tensorflow model
        if(config.get("SAVE_FINAL_WEIGHTS", False)):
            model.save_weights(os.path.join(config["CHECKPOINT_DIR"], "model-final.weights.h5"))
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
        file.create_dataset("n_signal_per_batch", data=train_generator.get_signals_per_batch())
        file.close()
        #do some clean-up
        close_data()
        train_generator.stop()
        val_generator.stop()
        toc2 = time()
    else:
        print("Running in test mode!")

    ### ----- PREDICTING/PLOTTING -----
    predictions = {}
    if(config["PERFORM_TEST"]):
        final_model = modelhandler.load_model(os.path.join(outdir, "model-checkpoints","model-final.keras"), config["MODEL"]) if args["test"] else model
        best_model_file = os.path.join(outdir, "model-checkpoints","model-checkpoint_best.weights.h5")
        if(os.path.isfile(best_model_file)):
            try:
                print("Recreating best model")
                best_model = modelhandler.create_dijet_model(config)
                best_model.load_weights(best_model_file)
            except Exception as e:
                print("Could not restore best model:", e)
                best_model = None
        else:
            best_model = None
        
        #We can't test signal vs background!
        # That info isn't available in a real world application
        if cross_validation_fold is not None:
            test_generator = get_kfold_data(cross_validation_fold, ["test"])[0]
        else:
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
            path = os.path.join(outdir, f"pred_{name.split(' ')[0]}.h5")
            file = h5py.File(path, 'w')
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
                extra_info = dict(jobinfo=loginfo, train_time=0, test_time=toc3-tic)
            else:
                extra_info = dict(jobinfo=loginfo, train_time=toc2-toc1, test_time=toc3-toc2)
            plotter.plot(outdir=outdir, model_name=config["MODELNAME"],
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
