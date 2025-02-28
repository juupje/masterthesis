#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, "/home/kd106458/thesis_devel/lib")
from time import time
import h5py
from DataLoaders import JetDataLoader as JDL
from DataLoaders.Discriminator import Discriminator
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
    return jl.JLogger(jobid=jobid, output_dir=outdir, comment=f"R{RUN_ID:d}")

"""
Total:
    - background: 1 000 000
    - signal: 100 000
SR:
    - background: 121350
    - signal: 75299
SB:
    - background: 212136+66650
    - signal: 11571 + 8842
"""

data = None
def open_data():
    global data
    if data is None:
        data = {"simulated":    h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_BG_SIM"])),
            "data_background":  h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_BG_DATA"])),
            "data_signal":      h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_SN_DATA"]))}
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
    k = config["CROSS_VALIDATION"]["K"]
    n_sim = int(config["N_SIMULATED"])
    n_bg = int(config["N_DATA_BACKGROUND"])
    n_sn = int(config["N_DATA_SIGNAL"])
    print(f"Simulated: {n_sim:d}/{data['simulated']['signal'].shape[0]}")
    print(f"Data\n\tBackground: {n_bg:d}/{data['data_background']['signal'].shape[0]}\n\tSignal: {n_sn:d}/{data['data_signal']['signal'].shape[0]}")
    data_seed = config.get("DATA_SEED") or config["SEED"]
    rng = np.random.default_rng(data_seed)
    print(f"SEED: {data_seed:d}")

    inputs = ["coords", "features", "mask"] if config["MODEL"] == "ParticleNet" else ["4mom", "features", "mask"]
    disc = Discriminator("jet_features/-1", lower=config["SR"][0], upper=config["SR"][1])
    decoder_inputs = get_decoder_inputs()

    #determine the slices of the k folds
    sim_slices = np.array([slice(int(n_sim/k*i), int(n_sim/k*(i+1))) for i in range(k)])
    bg_slices = np.array([slice(int(n_bg/k*i), int(n_bg/k*(i+1))) for i in range(k)])
    signal_per_slice = rng.multinomial(n_sn, np.ones(k)/k)
    signal_slice_bounds = np.cumsum(np.insert(signal_per_slice, 0, 0))
    sn_slices = np.array([slice(signal_slice_bounds[i], signal_slice_bounds[i+1]) for i in range(0,k)])
    print(signal_per_slice)
    #roll the slices according to the fold number
    sim_slices = np.roll(sim_slices, shift=-fold_nr, axis=0)
    bg_slices = np.roll(bg_slices, shift=-fold_nr, axis=0)
    sn_slices = np.roll(sn_slices, shift=-fold_nr, axis=0)

    #take the correct slides according to the split
    gens = []
    for split in splits:
        split_idx = config["CROSS_VALIDATION"][split]
        sim_slices_split = [sim_slices[idx] for idx in split_idx]
        bg_slices_split = [bg_slices[idx] for idx in split_idx]
        sn_slices_split = [sn_slices[idx] for idx in split_idx]
        print(f"{k}-Fold cross-validation with fold {fold_nr} and split {split}:")
        print("\tUsing simulated slices:", sim_slices_split)
        print("\tUsing background slices:", bg_slices_split)
        print("\tUsing signal slices:", sn_slices_split)
        gens.append(gen := JDL.IADDataLoaderV2(sim_bg=data["simulated"], data_sn=data["data_signal"], data_bg=data["data_background"],batch_size=config["BATCH_SIZE"],
                                        N_simulated=-1, N_background=-1, N_signal=-1, features=config["FEATURES"], seed=data_seed+split_idx[0], particles=config["N"],
                                        decoder_inputs=decoder_inputs,
                                        noise_features=config.get("NOISE_FEATURES", None), noise_param=config.get("NOISE_PARAM", (0,1)), noise_type=config.get("NOISE_TYPE", "normal"),
                                        inputs=inputs, njets=2, discriminator=disc, oversampling=(config["OVERSAMPLING"] if split!='test' else None),
                                        sim_bg_slice=sim_slices_split, data_bg_slice=bg_slices_split, data_sn_slice=sn_slices_split,
                                        include_true_labels=(split!="train")))
        print(f"{split} generator: using {len(gen)} steps")
    return gens

def get_data(test:bool=False):
    open_data()
    n_sim_t = int(config["N_SIMULATED_TRAIN"])
    n_bg_t = int(config["N_DATA_BACKGROUND_TRAIN"])
    n_sn_t = int(config["N_DATA_SIGNAL_TRAIN"])
    assert n_sn_t>=0, "Invalid signal ratio"

    n_sim_v = int(config["N_SIMULATED_VAL"])
    n_bg_v = int(config["N_DATA_BACKGROUND_VAL"])
    n_sn_v = int(config["N_DATA_SIGNAL_VAL"])

    data_seed = config.get("DATA_SEED") or config["SEED"]
    print(f"Simulated: {n_sim_t:d}/{data['simulated']['signal'].shape[0]}")
    print(f"Data\n\tBackground: {n_bg_t:d}/{data['data_background']['signal'].shape[0]}\n\tSignal: {n_sn_t:d}/{data['data_signal']['signal'].shape[0]}")
    print(f"SEED: {data_seed:d}")
    validate_data()

    inputs = ["coords", "features", "mask"] if config["MODEL"] == "ParticleNet" else ["4mom", "features", "mask"]
    disc = Discriminator("jet_features/-1", lower=config["SR"][0], upper=config["SR"][1])
    decoder_inputs = get_decoder_inputs()

    if(not test):
        train_generator = JDL.IADDataLoaderV2(sim_bg=data["simulated"], data_sn=data["data_signal"], data_bg=data["data_background"],batch_size=config["BATCH_SIZE"],
                                    N_simulated=n_sim_t, N_background=n_bg_t, N_signal=n_sn_t, features=config["FEATURES"], seed=data_seed, particles=config["N"],
                                    decoder_inputs=decoder_inputs,
                                    noise_features=config.get("NOISE_FEATURES", None), noise_param=config.get("NOISE_PARAM", (0,1)), noise_type=config.get("NOISE_TYPE", "normal"),
                                    inputs=inputs, njets=2, discriminator=disc, oversampling=config["OVERSAMPLING"],
                                    sim_bg_slice=slice(n_sim_t), data_bg_slice=slice(n_bg_t), data_sn_slice=slice(n_sn_t))
        val_generator = JDL.IADDataLoaderV2(sim_bg=data["simulated"], data_sn=data["data_signal"],data_bg=data["data_background"], batch_size=config["BATCH_SIZE"],
                                    N_simulated=n_sim_v, N_background=n_bg_v, N_signal=n_sn_v, features=config["FEATURES"], seed=data_seed+1, particles=config["N"],
                                    decoder_inputs=decoder_inputs,
                                    noise_features=config.get("NOISE_FEATURES", None), noise_param=config.get("NOISE_PARAM", (0,1)), noise_type=config.get("NOISE_TYPE", "normal"),
                                    inputs=inputs, njets=2, discriminator=disc, oversampling=config["OVERSAMPLING"],
                                    sim_bg_slice=slice(n_sim_t,n_sim_t+n_sim_v), data_bg_slice=slice(n_bg_t, n_bg_t+n_bg_v), data_sn_slice=slice(n_sn_t, n_sn_t+n_sn_v),
                                    include_true_labels=True)
        print(f"Train generator: {len(train_generator):d}\nVal generator: {len(val_generator):d}")
        print(f"{len(train_generator)} training steps and {len(val_generator)} validation steps")
        for key in data: data[key].close()
        return train_generator, val_generator
    else:
        factor = 1/config["FACTOR"]
        offset_sim, offset_bg, offset_sn = n_sim_t+n_sim_v, n_bg_t+n_bg_v, n_sn_t+n_sn_v
        test_generator = JDL.IADDataLoaderV2(sim_bg=data["simulated"], data_sn=data["data_signal"], data_bg=data["data_background"],batch_size=config["BATCH_SIZE"], particles=config["N"],
                                    N_simulated=int(n_sim_t*factor), N_background=int(n_bg_t*factor), N_signal=int(n_sn_t*factor), features=config["FEATURES"], seed=data_seed+2, do_shuffle=False,
                                    noise_features=config.get("NOISE_FEATURES", None), noise_param=config.get("NOISE_PARAM", (0,1)), noise_type=config.get("NOISE_TYPE", "normal"),
                                    inputs=inputs, njets=2, discriminator=disc, oversampling=False, decoder_inputs=decoder_inputs,
                                    sim_bg_slice=slice(offset_sim, offset_sim+int(n_sim_t*factor)), data_bg_slice=slice(offset_bg, offset_bg+int(n_bg_t*factor)), data_sn_slice=slice(offset_sn, offset_sn+int(n_sn_t*factor)))
        print(f"Test generator: {len(test_generator):d}")
        print(f"{len(test_generator)} test steps")
        for key in data: data[key].close()
        return test_generator

def get_test_data():
    separate_test_data = "PREPROCESSED_TEST_DATA_BG" in config
    if(separate_test_data):
        print("Using separate test data!")
        data_bg = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_TEST_DATA_BG"]), mode='r')
        data_sn = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_TEST_DATA_SN"]), mode='r')
        offset_bg, offset_sn = 0, 0
    else:
        print("Getting test data from simulated background and data signal!")
        data_bg = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_BG_SIM"]), mode='r')
        data_sn = h5py.File(os.path.join(config["DATA_DIR"], config["PREPROCESSED_DATA_SN_DATA"]), mode='r')
        if cross_validation_fold is not None:
            offset_bg, offset_sn = int(config["N_SIMULATED"]), int(config["N_DATA_SIGNAL"])
        else:
            offset_bg = int(config["N_SIMULATED_TRAIN"]+config["N_SIMULATED_VAL"])
            offset_sn = int(config["N_DATA_SIGNAL_TRAIN"]+config["N_DATA_SIGNAL_VAL"])
    l = ["coords", "features", "mask"] if config["MODEL"] == "ParticleNet" else ["4mom", "features", "mask"]
    jets = ["jet1", "jet2"]
    features = config["FEATURES"]
    if(features is None):
        l.remove("features")
    disc = Discriminator("jet_features/-1", lower=config["SR"][0], upper=config["SR"][1])
    #only select events in the Signal Region (SR)
    is_sr_background = disc.apply(data_bg)[offset_bg:]
    is_sr_signal = disc.apply(data_sn)[offset_sn:]
    #count how many of those events we have
    idx_bg = np.where(is_sr_background)[0]
    idx_sn = np.where(is_sr_signal)[0]
    #pick the required number of events
    n_test_bg = config["N_TEST_BACKGROUND"] if config["N_TEST_BACKGROUND"]>0 else np.infty
    n_test_sn = config["N_TEST_SIGNAL"] if config["N_TEST_SIGNAL"]>0 else np.infty
    n_bg, n_sn = min(idx_bg.shape[0], n_test_bg), min(idx_sn.shape[0], n_test_sn)
    print(f"Using {n_bg} background and {n_sn} signal test events")
    idx_bg = idx_bg[:n_bg]#np.random.choice(idx_bg, n_bg, replace=False)
    idx_sn = idx_sn[:n_sn]#np.random.choice(idx_sn, n_sn, replace=False)
    
    particles = config["N"]
    bg = {jet: {x:np.array(data_bg[jet][x][offset_bg:, :particles])[idx_bg] for x in l} for jet in jets}
    sn = {jet: {x:np.array(data_sn[jet][x][offset_sn:, :particles])[idx_sn] for x in l} for jet in jets}
    
    if config.get("EXTRA_DECODER_INPUTS",None) is not None:
        from DataLoaders import DecoderFeatures as DF
        decoder_inputs = [DF.get(name) for name in config["EXTRA_DECODER_INPUTS"]]
        if(len(decoder_inputs)>1):
            bg_decoder = np.concatenate((dec(bg) for dec in decoder_inputs),axis=-1)
            sn_decoder = np.concatenate((dec(sn) for dec in decoder_inputs),axis=-1)
        else:
            bg_decoder, sn_decoder = decoder_inputs[0](bg), decoder_inputs[0](sn)
    else:
        bg_decoder, sn_decoder = [], []
        
    if(features != 'all' and features is not None):
        #select the required features
        for jet in jets:
            bg[jet]["features"] = bg[jet]["features"][:,:,features]
            sn[jet]["features"] = sn[jet]["features"][:,:,features]
    if(config.get("NOISE_FEATURES", None)):
        from utils import noise_gen
        seed = config.get("DATA_SEED") or config["SEED"]
        
        for jet in jets:
            bg[jet]["features"] = np.concatenate((bg[jet]["features"], noise_gen.sample(config["NOISE_TYPE"], config.get("NOISE_PARAM", (0,1)), size=(*bg[jet]["features"].shape[:-1], 1), seed=seed)),axis=-1)
            sn[jet]["features"] = np.concatenate((sn[jet]["features"], noise_gen.sample(config["NOISE_TYPE"], config.get("NOISE_PARAM", (0,1)), size=(*sn[jet]["features"].shape[:-1], 1), seed=seed)),axis=-1)
    data_bg.close()
    data_sn.close()
    return [bg[jet][x] for jet in jets for x in l]+bg_decoder, [sn[jet][x] for jet in jets for x in l]+sn_decoder, idx_bg, idx_sn

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
    logger = setup_logging()
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
        #weights = {0:1, 1:1/train_generator.get_signal_ratio()}
        #print("WEIGHTS:", weights)
        result = model.fit(train_generator,
            #class_weight=weights,
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
        bg, sn, idx_bg, idx_sn = get_test_data()
        print(sn[0].shape, bg[0].shape)
        if(config.get("TEST_SR_VS_BG",True)):
            #get 'real' sr vs bg data
            if cross_validation_fold is not None:
                test_generator = get_kfold_data(cross_validation_fold, ["test"])[0]
            else:
                test_generator = get_data(test=True)
            iad_labels = test_generator.labels[:,1]
            true_labels = test_generator.true_labels[:,1]
            is_cr = iad_labels==0
            print(f"CR size: {np.sum(is_cr):d}")
            print(f"SR size: {np.sum(~is_cr):d}")
            test_data = [test_generator.data[jet][s] for jet in test_generator.jets for s in test_generator.inputs]
        else:
            test_data = None
        for model, name in zip([final_model, best_model], ["final", f"best ({config['MONITOR']})"]):
            if(model is None): continue #to catch non-existent best model
            p_bg = model.predict(bg, batch_size=config["BATCH_SIZE"], verbose=config["VERBOSITY"])
            p_sn = model.predict(sn, batch_size=config["BATCH_SIZE"], verbose=config["VERBOSITY"])
            pred = np.concatenate((p_bg,p_sn),axis=0)
            labels = np.concatenate([np.zeros(p_bg.shape[0], dtype=np.int8),np.ones(p_sn.shape[0], dtype=np.int8)])
            idx = np.concatenate([idx_bg, idx_sn], axis=0)
            path = os.path.join(outdir, f"pred_{name.split(' ')[0]}.h5")
            file = h5py.File(path, 'w')
            group = file.create_group("s_vs_b")
            group.create_dataset("pred", data=pred)
            group.create_dataset("label", data=labels)
            group.create_dataset("data_idx", data=idx)

            if(test_data is not None):
                pred = model.predict(test_data, batch_size=config["BATCH_SIZE"], verbose=config["VERBOSITY"])
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
