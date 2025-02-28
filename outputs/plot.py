import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import steganologger, utils

def extract_at_value(arr:np.ndarray, value:float, others:list, thres=1e-3):
    for other in others:
        if(other.shape!=arr.shape):
            raise ValueError(f"Lengths of arrays do not match {len(other):d} was expected to be {len(arr):d}")
    diff = np.abs(arr-value)
    idx = np.argmin(diff)
    if(diff[idx]<thres):
        return idx, [other[idx] for other in others]
    else:
        return None

def plot(outdir:str, logger, training_data_file:str=None, model_name:str="Model", prediction_files:dict=None,config:dict=None,
         extra_info:dict=None, plot_lr:bool=False, plot_score_hist:bool=False, store_format:str='png', upload:bool=False):
    assert store_format in ["png", "pdf", "svg", "pgf"], "Invalid file format"
    if(store_format == "pgf"):
        import matplotlib
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'font.size' : 11,
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    if upload:
        from onedrive import onedrive
        od_handler = onedrive.OneDriveHandler()
    if training_data_file is not None and len(training_data_file)>0:
        #plot the training curves
        file = h5py.File(training_data_file, 'r')
        loss = np.array(file["loss"])
        val_loss = np.array(file["val_loss"])
        acc = np.array(file["accuracy"])
        val_acc = np.array(file["val_accuracy"])
        if(plot_lr):
            lr = np.array(file["lr"])

        fig, ((ax_t, ax_v), (ax_l_sim, ax_l_data), (ax_l_bg, ax_l_sn)) = plt.subplots(3,2, sharex=True, figsize=(10,8))
        x = np.arange(1,len(loss)+1)

        #Normal loss and accuracy
        ax_t.plot(x,loss, color='C0',label="Training loss")
        ax_v.plot(x,val_loss, color='C0', label='Validation loss')
        ratio = (config["N_DATA_BACKGROUND_VAL"]+config["N_DATA_SIGNAL_VAL"])/config["N_SIMULATED_VAL"]
        ax_v.plot(x,(1*np.array(file["val_sim_loss"])+ratio*np.array(file["val_data_loss"]))/(1+ratio), color='C0', linestyle='--')
        ax_t2, ax_v2 = ax_t.twinx(), ax_v.twinx()
        ax_t2.plot(x,acc, color='C1', label="Training accuracy")
        ax_v2.plot(x,val_acc, color='C1', label='Validation accuracy')

        ax_t.set_ylabel("Loss (cat. cross-entropy)", color='C0')
        ax_v.set_ylabel("Loss (cat. cross-entropy)", color='C0')
        ax_t2.set_ylabel("Accuracy", color='C1')
        ax_v2.set_ylabel("Accuracy", color='C1')
        ax_t.set_title("Training")
        ax_v.set_title("Validation")
        ax_t.grid()
        ax_v.grid()

        to_plot = { "val_sim_loss": {"name": "Loss (Simulated only)", "axis": ax_l_sim},
                    "val_data_loss": {"name": "Loss (Data only)", "axis": ax_l_data},
                    "val_bg_loss": {"name": "Loss (Background only)", "axis": ax_l_bg},
                    "val_sn_loss": {"name": "Loss (Signal only)", "axis": ax_l_sn}}
        coloriter = utils.colors.ColorIter()
        for key, val in to_plot.items():
            val["axis"].plot(x, np.array(file[key]), color=next(coloriter))
            val["axis"].set_ylabel("Loss (cat. cross-entropy)")
            val["axis"].set_xlabel("Epoch")
            val["axis"].grid()
            val["axis"].set_title(val["name"])
            print(val["name"], file[key][3])

        fig.tight_layout()
        fig.suptitle(f"Training of {model_name}")
        trainplot = os.path.join(outdir,f"training.{store_format}")
        print("Saving plot: " + trainplot)
        logger.log_figure(fig, trainplot, dpi=250)
        file.close()
        if(config is not None):
            steganologger.encode(trainplot, dict(config=config, extra_info=extra_info), overwrite=True)

        if(plot_lr):
            fig,ax = plt.subplots()
            ax.plot(np.arange(len(lr))+1, lr, label='lr')
            ax.set_xlabel("epoch")
            ax.set_ylabel("learning rate")
            ax.set_title(f"Learning rate of {model_name:s}")
            ax.grid()
            fig.legend()
            lr_file = os.path.join(outdir, f"learning_rate.{store_format}")
            logger.log_figure(fig, lr_file, data_file=training_data_file, dpi=250)

    if(prediction_files is not None and len(prediction_files.keys())>0):
        old = np.seterr(divide='ignore')
        #plot the ROC curve
        fig, ax = plt.subplots(figsize=(4,3.5))
        results = {}
        coloriter = utils.colors.ColorIter()
        for i,key in enumerate(prediction_files):
            file = h5py.File(prediction_files[key], 'r')
            pred = np.array(file["pred"])
            y_true = np.array(file["label"])
            file.close()
            y_pred = pred[:,1]
            is_nan = np.isnan(y_pred)
            y_pred = y_pred[~is_nan]
            y_true = y_true[~is_nan]
            print(f"Predicted {np.sum(is_nan)} nan's")

            fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
            res1 = extract_at_value(tpr, 0.2, others=[fpr,thres], thres=0.01)
            if(res1):  _, [fpr_20, thres_20] = res1
            res2 = extract_at_value(tpr, 0.4, others=[fpr,thres], thres=0.01)
            if(res2):  _, [fpr_40, thres_40] = res2
            auroc = roc_auc_score(y_true, y_pred)
            results[key] = {"@eps_s=0.2": {"1/eps_b": 1/fpr_20, "thres": thres_20} if res1 else {},
                            "@eps_s=0.4": {"1/eps_b": 1/fpr_40, "thres": thres_40} if res2 else {},
                            "auroc": auroc}

            color = next(coloriter)
            inv_fpr = 1/fpr
            inv_fpr[np.isinf(inv_fpr)] = np.nan
            ax.plot(tpr, 1/fpr, color=color,label=key)
            #ax.text(0.7,0.8-i*0.05, f"AUROC={auroc:.4f}", color=color, transform=ax.transAxes)
        # random classifier
        np.seterr(**old)
        x = np.linspace(0.001,1,100)
        ax.plot(x, 1/x, 'k--')

        #Make it pretty
        ax.set_yscale('log')
        ax.set_xlim(0,1)
        ax.set_ylim(1,1e5)
        ax.set_xlabel(r"$\epsilon_S$")
        ax.set_ylabel(r"$1/\epsilon_B$")
        ax.grid()
        ax.legend()
        #fig.suptitle(f"ROC curve for {model_name}")
        utils.format_floats(results, "{:.4f}".format)
        rocfile = os.path.join(outdir,f"roc.{store_format}")
        logger.log_figure(fig, rocfile, data_file=", ".join(list(prediction_files.values())), dpi=250, bbox_inches='tight')
        if(store_format=='pgf'):
            fig.savefig(os.path.join(outdir, "roc.png"), dpi=250, bbox_inches='tight')
        if(config is not None):
            steganologger.encode(rocfile, dict(config=config, results=results, extra_info=extra_info), overwrite=True)
        if(upload):
            od_handler.upload(rocfile, name=f"rocs.{store_format}", to_path="Documenten/thesis/plots/training")
        #Plot the score distribution
        if(plot_score_hist):
            fig, ax = plt.subplots()
            bins = {}
            colors = utils.colors.ColorIter()
            for i,key in enumerate(prediction_files):
                file = h5py.File(prediction_files[key], 'r')
                pred = np.array(file["pred"])
                y_true = np.array(file["label"])
                file.close()
                is_nan = np.isnan(pred[:,1])
                pred = pred[~is_nan]
                y_true = y_true[~is_nan]
                color = next(colors)
                is_signal = y_true==1
                _,bins["signal"],_ = ax.hist(pred[is_signal,1], bins=bins.get("signal", 100), color=color, density=True, histtype='step',label=f"true signal / {key}")
                _,bins["background"],_ = ax.hist(pred[~is_signal,1], bins=bins.get("background", 100), color=color, density=True, histtype='step', linestyle='dashed', label=f"true background / {key}")
            ax.set_yscale("log")
            ax.set_xlabel("Score")
            ax.set_ylabel("Relative frequency")
            ax.grid()
            ax.legend()
            fig.suptitle(f"Score distribution for {model_name}")
            scorefile = os.path.join(outdir, "score_hist.png")
            logger.log_figure(fig, scorefile, data_file=", ".join(list(prediction_files.values())), dpi=250)
            if(config is not None):
                steganologger.encode(scorefile, dict(config=config, extra_info=extra_info), overwrite=True)

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
    parser.add_argument("--plot_score", help="Plot score distribution", action='store_true')
    parser.add_argument("--format", "-f", help="The file format", choices=["pdf", "svg", "png", "pgf"], default='png')
    parser.add_argument("--upload", "-u", help="Should the ROC plot be uploaded?", action="store_true")
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
            plot_lr=args["plot_lr"], plot_score_hist=args["plot_score"], store_format=args['format'], upload=args["upload"])
    else:
        plot(outdir, jl.JLogger(), args["training"], model_name=config["MODELNAME"],
            prediction_files={args["label"][i]: predict_files[i] for i in range(min(len(predict_files), len(args["label"])))}, config=config,
            plot_lr=args["plot_lr"], plot_score_hist=args["plot_score"], store_format=args['format'], upload=args["upload"])