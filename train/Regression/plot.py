import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import steganologger, utils

def plot(outdir:str, logger, training_data_file:str=None, model_name:str="Model", prediction_files:dict=None,config:dict=None,
         extra_info:dict=None, plot_lr:bool=False, store_format:str='png'):
    assert store_format in ["png", "pdf", "svg"], "Invalid file format"
    if training_data_file is not None:
        if not os.path.exists(training_data_file):
            print("Training file doesn't exist. Skipping...")
        else:
            # ============ PLOT TRAINING CURVES ============ #
            file = h5py.File(training_data_file, 'r')
            loss = np.array(file["loss"])
            val_loss = np.array(file["val_loss"])
            if(plot_lr):
                lr = np.array(file["lr"])

            fig, (ax_t, ax_v) = plt.subplots(1,2, sharex=True, figsize=(10,6))
            x = np.arange(1,len(loss)+1)

            #Normal loss and accuracy
            ax_t.plot(x,loss, color='C0',label="Training loss")
            ax_v.plot(x,val_loss, color='C0', label='Validation loss')
            ax_t.set_ylabel("Loss (cat. cross-entropy)", color='C0')
            ax_v.set_ylabel("Loss (cat. cross-entropy)", color='C0')
            ax_t.set_title("Training")
            ax_v.set_title("Validation")
            ax_t.grid()
            ax_v.grid()

            fig.tight_layout()
            fig.suptitle(f"Training of {model_name}")
            trainplot = os.path.join(outdir,f"training.{store_format}")
            print("Saving plot: " + trainplot)
            logger.log_figure(fig, trainplot, dpi=250)
            if(config is not None):
                steganologger.encode(trainplot, dict(config=config, extra_info=extra_info), overwrite=True)

            # ============ PLOT LEARNING RATE SCHEDULE ============ #
            if(plot_lr):
                fig,ax = plt.subplots()
                lr = np.array(file["lr"])
                ax.plot(np.arange(len(lr))+1, lr, label='lr')
                lr_batch = np.array(file["learning_rate"])
                lr_batch = lr_batch.flatten()
                ax.plot(np.linspace(0, len(lr)+1, lr_batch.shape[0], endpoint=False), lr_batch, label='per batch')
                ax.set_xlabel("epoch")
                ax.set_ylabel("learning rate")
                ax.set_title(f"Learning rate of {model_name:s}")
                ax.grid()
                fig.legend()
                lr_file = os.path.join(outdir, f"learning_rate.{store_format}")
                logger.log_figure(fig, lr_file, data_file=training_data_file, dpi=250)
            file.close()

    # ============ PLOT PREDICTION CURVES ============ #
    if(prediction_files is not None and len(prediction_files.keys())>0):
        #plot the ROC curve
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,6))
        ax1:plt.Axes = ax1
        results = {}
        coloriter = utils.colors.ColorIter()
        target_color = next(coloriter)
        for i,key in enumerate(prediction_files):
            file = h5py.File(prediction_files[key], 'r')
            reco = np.array(file["reco"])
            target = np.array(file["target"])
            file.close()
            #calculate the loss!
            results[key] = {"loss":mean_squared_error(target, reco)}

            bins = np.linspace(0, 3.78, 100)
            color = next(coloriter)
            reco, target = np.log10(reco), np.log10(target)
            ax1.hist(reco,   bins=bins, histtype='step', linestyle='-', color=color, label=f"{key}/reco")
            ax2.hist((target-reco), bins=100, histtype='step', color=color)
        #plot the target only once (we assume that all models have the same target)
        ax1.hist(target, bins=bins, histtype='step', linestyle='--', color=target_color, label=f"target")
        #what would the reconstruction error look like if you randomly sampled from the target distribution?
        #reco_random = target.copy()
        #np.random.shuffle(reco_random)
        #ax2.hist((target-reco_random), bins=100, histtype='step', linestyle='--', color=target_color)

        ax1.set_xlabel(r"$log(m_{jj}/\mathrm{GeV})$")
        ax1.set_ylabel("Freq.")
        ax2.set_xlabel(r"$\hat{m}_{target}-\hat{m}_{reco}$")
        ax2.set_ylabel("Freq.")

        major_ticks = np.array([1000, 2000, 3000, 4000, 5000,6000])
        minor_ticks = np.array([1500, 2500, 3500, 4500, 5500])
        major_tick_locations = np.interp(np.log10(major_ticks), np.linspace(*ax1.get_xlim(), 20), np.linspace(0,1,20))
        minor_tick_locations = np.interp(np.log10(minor_ticks), np.linspace(*ax1.get_xlim(), 20), np.linspace(0,1,20))
        #add second axis to indicate true masses
        _ax1 = ax1.twiny()
        fig.subplots_adjust(bottom=0.2)

        _ax1.xaxis.set_ticks_position("bottom")
        _ax1.xaxis.set_label_position("bottom")
        _ax1.spines["bottom"].set_position(("axes", -0.15))
        _ax1.set_frame_on(True)
        _ax1.patch.set_visible(False)
        for sp in _ax1.spines:
            _ax1.spines[sp].set_visible(False)
        _ax1.spines["bottom"].set_visible(True)
        _ax1.set_xticks(major_tick_locations, minor=False)
        _ax1.set_xticklabels(major_ticks, minor=False)
        _ax1.set_xticks(minor_tick_locations, minor=True)
        _ax1.set_xlabel(r"$m_{jj}$ [GeV]")

        ax1.legend()
        fig.suptitle(f"Reconstruction distribution for {model_name}")
        utils.format_floats(results, "{:.4f}".format)
        rocfile = os.path.join(outdir,f"reco.{store_format}")
        logger.log_figure(fig, rocfile, data_file=", ".join(list(prediction_files.values())), dpi=250)
        if(config is not None):
            steganologger.encode(rocfile, dict(config=config, results=results, extra_info=extra_info), overwrite=True)

if __name__=="__main__":
    import jlogger as jl
    import utils
    import argparse

    parser = argparse.ArgumentParser("Plotter of model training output")
    parser.add_argument("--config", "-c", help="Path of the configuration file (.json or .py)", default='config.json', type=str)
    parser.add_argument("--outdir", "-o", help="Path of the output directory", type=str)
    parser.add_argument("--prediction", "-p", help="Path of the prediction file", nargs='*', default=['pred_final.h5', 'pred_best.h5'], type=str)
    parser.add_argument("--label", "-l", help="Labels of the prediction files", nargs='*', default=['final','best'], type=str)
    parser.add_argument("--training", "-t", help="Path of the training log", default='training_stats.h5', type=str)
    parser.add_argument("--relative", "-r", help="Treat all file paths as relative to 'outdir'", action='store_true')
    parser.add_argument("--plot_lr", help="Plot learning rate", action='store_true')
    parser.add_argument("--format", "-f", help="The file format", choices=["pdf", "svg", "png"], default='png')
    args = vars(parser.parse_args())

    outdir = os.path.dirname(__file__)
    if(args["outdir"]):
        outdir = os.path.join(outdir, args["outdir"])
    print(f"Using output dir {outdir:s}")
    config_file = os.path.join(outdir, args["config"]) if args["relative"] else args["config"]
    config = utils.parse_config(config_file)
    print(f"Using config file {config_file:s}")

    predict_files = args["prediction"]
    assert len(args["label"])==len(args["prediction"]), "Arguments 'label' and 'prediction' should have same number of values"
    if(args["relative"]):
        plot(outdir, jl.JLogger(), os.path.join(outdir, args["training"]) if args["training"] else None, model_name=config["MODELNAME"],
            prediction_files={args["label"][i]: os.path.join(outdir, predict_files[i]) for i in range(min(len(predict_files), len(args["label"])))}, config=config,
            plot_lr=args["plot_lr"], store_format=args['format'])
    else:
        plot(outdir, jl.JLogger(), args["training"], model_name=config["MODELNAME"],
            prediction_files={args["label"][i]: predict_files[i] for i in range(min(len(predict_files), len(args["label"])))}, config=config,
            plot_lr=args["plot_lr"], store_format=args['format'])