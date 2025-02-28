import matplotlib.pyplot as plt
import h5py
import os, glob
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import steganologger, utils
import re

def plot_epochs(outdir:str, logger, model_name:str="Model",config:dict=None, main_script="main.py",
         extra_info:dict=None, start:int=0, end:int=-1, checkpoint_dir:str="model-checkpoints",
          load_prediction:str=None, save_predictions:bool=False, store_format:str='png'):
    assert all([x in ["png", "svg", "html"] for x in store_format.split("+")]), "Invalid file format"
    
    # ============ PLOT ROC CURVES ============ #
    # Get test data
    main = utils.imports.import_mod("main", main_script)
    main.config = config

    # find the models (one per epoch)
    if(end == -1):
        end = np.infty

    if(load_prediction):
        pred_file = h5py.File(os.path.join(outdir, load_prediction), 'r')
        epochs = np.array(pred_file["epochs"])
        model_files = [None]*epochs.shape[0]
        print("Using stores predictions. Background:",pred_file["background"].shape, "Signal:", pred_file["signal"].shape)
        print("Epochs:", epochs)
    elif(save_predictions):
        from models import modelhandler
        bg, sn, idx_bg, idx_sn = main.get_test_data()
        assert np.all(idx_bg[:-1] < idx_bg[1:])
        assert np.all(idx_sn[:-1] < idx_sn[1:])
        print(sn[0].shape, bg[0].shape)

        model_files, epoch_idxs = [], []
        for file in glob.glob(os.path.join(outdir, checkpoint_dir, "*-*.h5")):
            match = re.match(r".*/.*-(\d+)\.h5", file)
            if match:
                if(start <= int(match.group(1)) <= end):
                    model_files.append(file)
                    epoch_idxs.append(int(match.group(1)))
        idx = np.argsort(epoch_idxs)
        model_files, epochs = np.array(model_files)[idx], np.array(epoch_idxs)[idx]
        pred_file = h5py.File(os.path.join(outdir, "test_predictions.h5"), 'w')
        pred_file.create_dataset("background", shape=(epochs.shape[0], bg[0].shape[0]), dtype=np.float32, compression='gzip')
        pred_file.create_dataset("signal", shape=(epochs.shape[0], sn[0].shape[0]), dtype=np.float32, compression='gzip')
        pred_file.create_dataset("epochs", data=epochs)
    # evaluate each model
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)
    results = {}
    old = np.seterr(divide='ignore')
    coloriter = utils.colors.ColorIter()
    lines = []
    target_tprs, target_fprs = [0.2,0.4], [1e-3, 1e-4]
    res_inv_fprs, res_tprs, res_auc = [], [], []
    for i , (mfile, epoch) in enumerate(zip(model_files, epochs)):
        name = f"Epoch {epoch}"
        print(name, ":", mfile)
        if(load_prediction):
            p_bg = np.array(pred_file["background"][i])
            p_sn = np.array(pred_file["signal"][i])
        else:
            model = modelhandler.load_model(mfile, model_name)
            p_bg = model.predict(bg, batch_size=config["BATCH_SIZE"], verbose=1)[:,1]
            p_sn = model.predict(sn, batch_size=config["BATCH_SIZE"], verbose=1)[:,1]
            if(save_predictions):
                pred_file["background"][i,...] = p_bg
                pred_file["signal"][i,...] = p_sn

        y_pred = np.concatenate((p_bg,p_sn),axis=0)
        y_true = np.concatenate([np.zeros(p_bg.shape[0], dtype=np.int8),np.ones(p_sn.shape[0], dtype=np.int8)])
        is_nan = np.isnan(y_pred)
        y_pred = y_pred[~is_nan]
        y_true = y_true[~is_nan]
        print(f"Predicted {np.sum(is_nan)} nan's")
        fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
        results[name] = {}
        res_inv_fpr = []
        for target_tpr in target_tprs:
            res = utils.calc.extract_at_value(tpr, target_tpr, others=[fpr,thres], thres=0.01)
            if(res):  _, [_fpr, _thres] = res
            results[name][f"@eps_s={target_tpr:.2f}"] = {"1/eps_b":1/_fpr, "thres": _thres} if res else {}
            res_inv_fpr.append(1/_fpr)
        res_tpr = []
        for target_fpr in target_fprs:
            res = utils.calc.extract_at_value(fpr, target_fpr, others=[tpr,thres], thres=0.01)
            if(res):  _, [_tpr, _thres] = res
            results[name][f"@1/eps_b={1/target_fpr:.1e}"] = {"eps_s":_tpr, "thres": _thres} if res else {}
            res_tpr.append(_tpr)
        auroc = roc_auc_score(y_true, y_pred)
        results[name]["auroc"] = auroc

        res_tprs.append(res_tpr)
        res_inv_fprs.append(res_inv_fpr)
        res_auc += auroc

        color = next(coloriter)
        inv_fpr = 1/fpr
        inv_fpr[np.isinf(inv_fpr)] = np.nan
        lines.append(ax1.plot(tpr, 1/fpr, color=color,label=name))
    # random classifier
    np.seterr(**old)
    x = np.linspace(0.001,1,100)
    ax1.plot(x, 1/x, 'k--')

    #Make it pretty
    ax1.set_yscale('log')
    ax1.set_xlim(0,1)
    ax1.set_ylim(1,1e5)
    ax1.set_xlabel(r"$\epsilon_S$")
    ax1.set_ylabel(r"$1/\epsilon_B$")
    ax1.grid()
    ax1.legend(loc='center left', ncols=2, bbox_to_anchor=(1,0.5))
    
    # ======= plot progression of 1/fpr =======
    res_inv_fprs = np.array(res_inv_fprs)
    coloriter.rewind()
    for i, x in enumerate(target_tprs):
        ax2.plot(epochs, res_inv_fprs[:,i], color=next(coloriter), label=f"$\\epsilon_S={x:.1f}$")
    ax2.set_ylabel(r"$1/\epsilon_B$")
    ax2.set_title("Background rejection rate")

    # ======= plot progression of tpr =========
    res_tprs = np.array(res_tprs)
    coloriter.rewind()
    for i, x in enumerate(target_fprs):
        ax3.plot(epochs, res_tprs[:,i], color=next(coloriter), label=rf"$1/\epsilon_B={1/x:.0e}$")
    ax3.set_ylabel(r"$\epsilon_S$")
    ax3.set_title("Signal efficiency")

    for ax in (ax2,ax3):
        ax.grid()
        ax.legend()
        ax.set_xlabel("Epoch")

    fig.suptitle(f"ROC curve for {model_name}")
    utils.formatting.format_floats(results, "{:.4f}".format)
    for format in store_format.split("+"):
        rocfile = os.path.join(outdir,f"roc-epochs.{format}")
        if(format=="html"):
            import mpld3
            from mpld3 import plugins
            interactive_legend = plugins.InteractiveLegendPlugin(lines, [f"Epoch {idx}" for idx in epochs], start_visible=False)
            plugins.connect(fig, interactive_legend)
            mpld3.save_html(fig, rocfile)
        else:
            logger.log_figure(fig, rocfile, data_file=os.path.join(outdir, checkpoint_dir, "*-*.h5"), dpi=250)
            if(config is not None):
                steganologger.encode(rocfile, dict(config=config, results=results, extra_info=extra_info), overwrite=True)

if __name__=="__main__":
    import jlogger as jl
    import utils
    import argparse

    parser = argparse.ArgumentParser("Plotter of model training output")
    parser.add_argument("--config", "-c", help="Path of the configuration file (.json or .py)", default='config.json', type=str)
    parser.add_argument("--main_script", "-m", help="Path of the main script", default='main.m.py', type=str)
    parser.add_argument("--outdir", "-o", help="Path of the output directory", type=str)
    parser.add_argument("--relative", "-r", help="Treat all file paths as relative to 'outdir'", action='store_true')
    parser.add_argument("--format", "-f", help="The file format", default='png')
    parser.add_argument("--start", "-s", help="Start epoch", default=0, type=int)
    parser.add_argument("--end", "-e", help="End epoch", default=-1, type=int)
    parser.add_argument("--store", "-S", help="Store prediction", action='store_true')
    parser.add_argument("--load", "-l", help="Load from file", type=str)
    group = parser.add_mutually_exclusive_group()
    group2 = group.add_argument_group()
    group2.add_argument("--n_signal", help="Number of signal events", type=int)
    group2.add_argument("--n_background", help="Number of background events", type=int)
    group.add_argument("--n_data", "-n", help="Number of signal and background events")
    args = vars(parser.parse_args())

    outdir = os.path.dirname(__file__)
    if(args["outdir"]):
        outdir = os.path.join(outdir, args["outdir"])
    print(f"Using output dir {outdir:s}")
    config_file = os.path.join(outdir, args["config"]) if args["relative"] else args["config"]
    print(f"Using config file {config_file:s}")
    config = utils.configs.parse_config(config_file)
    if(args["n_data"]):
        config["N_TEST_SIGNAL"] = int(args["n_data"])
        config["N_TEST_BACKGROUND"] = int(args["n_data"])
    else:
        if(args["n_signal"]):
            config["N_TEST_SIGNAL"] = int(args["n_signal"])
        if(args["n_background"]):
            config["N_TEST_BACKGROUND"] = int(args["n_background"])
    plot_epochs(outdir, jl.JLogger(), model_name=config["MODELNAME"], config=config, main_script=args["main_script"], store_format=args['format'], start=args["start"], end=args["end"],
                save_predictions=args["store"], load_prediction=args["load"])