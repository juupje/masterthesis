import matplotlib.pyplot as plt
import h5py
import os, glob
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import steganologger, utils
from functools import reduce
from utils import shuffling
def plot(outdir:str, model_name:str="Model", prediction_files:dict=None,config:dict=None, extra_info:dict=None,
         plot_score_hist:bool=False, store_format:str='png', save:str=None, logx:bool=False, upload:list=[]):
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
    if len(upload)>0:
        from onedrive import onedrive
        od_handler = onedrive.OneDriveHandler()
        upload_fail = False
    if(prediction_files is not None and len(prediction_files.keys())>0):
        old = np.seterr(divide='ignore')
        if type(save)==list and 'all' in save: save = 'all'
        elif save is not None and type(save) != list: save = [save]
        #plot the ROC curve
        fig, ax = plt.subplots()
        results = {}
        colors = utils.colors.ColorIter()
        count = 0
        save_roc, save_pred, save_mean = False, False, False
        if(save=='all' or 'roc' in save):
            savefile = h5py.File(os.path.join(outdir, "test-roc.h5"), 'w')
            save_roc = True
        if(save=='all' or 'pred' in save):
            savefile_pred = h5py.File(os.path.join(outdir, "test-pred.h5"), 'w')
            save_pred = True
        if(save=='all' or 'mean' in save):
            savefile_mean = h5py.File(os.path.join(outdir, "test-pred-mean.h5"), 'w')
            save_mean = True

        for i,key in enumerate(prediction_files.keys()):
            #collect all the prediction files
            files = glob.glob(prediction_files[key], recursive=False)
            labels, y_preds = [], []
            print(files)
            if save_pred: savefile_pred.create_group(key)
            for file in files:
                h5pyfile = h5py.File(file, mode='r')
                idx = np.array(h5pyfile["data_idx"])
                y_pred, y_true = np.array(h5pyfile["pred"][:,1]), np.array(h5pyfile["label"])
                shuffling.do_the_unshuffle(y_pred, idx)
                shuffling.do_the_unshuffle(y_true, idx)
                labels.append(y_true)
                y_preds.append(y_pred)
                h5pyfile.close()
            assert all(np.all(labels[i]==labels[0]) for i in range(1,len(labels))), "Not all labels match"
            stack = np.stack(y_preds)
            print(np.corrcoef(stack, rowvar=True))
            is_nan = reduce(lambda a,b: a | b, (np.isnan(y_pred) for y_pred in y_preds))
            print(f"Predicted {np.sum(is_nan)} nan's")
            y_true = labels[0][~is_nan]
            if save_pred:
                for j,y_pred in enumerate(y_preds):
                    dataset = savefile_pred[key].create_dataset(f"{j}", data=y_pred)
                    dataset.attrs["file"] = files[j]
                savefile_pred[key].create_dataset("labels", data=labels[0])
            
            color = next(colors)
            y_pred = np.mean(stack, axis=0)
            if save_mean:
                dataset = savefile_mean.create_dataset(key, data=y_pred)
                dataset.attrs["files"] = files
                if i == 0:
                    savefile_mean.create_dataset("labels", data=y_true)
                else:
                    assert np.all(np.array(savefile_mean["labels"])==y_true), "Labels do not match"

            results[key] = {}
            fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
            if(save_roc):
                savefile.create_dataset(key, data=np.stack((fpr,tpr,thres),axis=1))

            for eps_s in (0.2,0.3,0.4,0.5):
                res = utils.calc.extract_at_value(tpr, eps_s, others=[fpr,thres], thres=0.01)
                if(res):  _, [fpr1, thres1] = res
                results[key][f"@eps_s={eps_s:.1f}"] = {"1/eps_b": 1/fpr1, "thres": thres1} if res else {}
            for eps_b in (1e-3,1e-4):
                res = utils.calc.extract_at_value(fpr, eps_b, others=[tpr,thres], thres=0.01)
                if(res):  _, [_tpr, _thres] = res
                results[key][f"@1/eps_b={1/eps_b:.1e}"] = {"eps_s": _tpr, "thres": _thres} if res else {}
            auroc = roc_auc_score(y_true, y_pred)
            results[key]["auroc"] = auroc

            inv_fpr = 1/fpr
            inv_fpr[np.isinf(inv_fpr)] = np.nan
            ax.plot(tpr, 1/fpr, color=color,label=f"{key} - AUC={auroc:.4f}")
            for k,x in enumerate(y_preds):
                fpr, tpr, thres = roc_curve(y_true, x, pos_label=1)
                ax.plot(tpr, 1/fpr, color=color, alpha=0.3)
                if save == 'all' or 'singles' in save:
                    savefile.create_dataset(f"singles/{key}/{k}", data=np.stack((fpr,tpr,thres),axis=1))
        
        if(save_roc): savefile.close()
        if(save_pred): savefile_pred.close()
        if(save_mean): savefile_mean.close()
        # random classifier
        np.seterr(**old)
        x = np.linspace(0.001,1,100)
        ax.plot(x, 1/x, 'k--')

        #Make it pretty
        ax.set_yscale('log')
        if(logx):
            ax.set_xscale('log')
            ax.set_xlim(1e-3,None)
        else:
            ax.set_xlim(0,1)
        ax.set_ylim(1,1e5)
        ax.set_xlabel(r"$\epsilon_S$")
        ax.set_ylabel(r"$1/\epsilon_B$")
        ax.grid()
        ax.legend()
        fig.suptitle(f"ROC curve for {model_name} ensemble")
        utils.formatting.format_floats(results, "{:.4f}".format)
        rocfile = os.path.join(outdir,f"roc.{store_format}")
        fig.savefig(rocfile, dpi=250)
        if(config is not None):
            steganologger.encode(rocfile, dict(config=config, results=results, extra_info=extra_info), overwrite=True)
            if("roc" in upload and not upload_fail):
                resp = od_handler.upload(rocfile, f"R{config['RUN_ID']}_{config['TAG']}_{config['NAME']}-roc-ens.{store_format}", to_path=f"Documenten/thesis/plots/{config['JOBNAME']}")
                upload_fail = resp.status_code//100!=2
        #Plot the score distribution
        if(plot_score_hist):
            fig, ax = plt.subplots()
            bins = {}
            colors = utils.colors.ColorIter()
            for i,key in enumerate(prediction_files):
                files = glob.glob(prediction_files[key], recursive=True)
                labels, y_preds = [], []
                print(files)
                for file in files:
                    h5pyfile = h5py.File(file, mode='r')
                    idx = np.array(h5pyfile["data_idx"])
                    y_pred, y_true = np.array(h5pyfile["pred"][:,1]), np.array(h5pyfile["label"])
                    shuffling.do_the_unshuffle(y_pred, idx)
                    shuffling.do_the_unshuffle(y_true, idx)
                    labels.append(y_true)
                    y_preds.append(y_pred)
                    h5pyfile.close()
                assert all(np.all(labels[i]==labels[0]) for i in range(1,len(labels))), "Not all labels match"
                is_nan = reduce(lambda a,b: a | b, (np.isnan(y_pred) for y_pred in y_preds))
                y_pred = np.max(np.stack(y_preds), axis=0)
                y_true = labels[0][~is_nan]
                color = next(colors)
                is_signal = y_true==1
                _,bins["signal"],_ = ax.hist(y_pred[is_signal], bins=100, color=color, density=True, histtype='step',label=f"true signal / {key}")
                _,bins["background"],_ = ax.hist(y_pred[~is_signal], bins=100, color=color, density=True, histtype='step', linestyle='dashed', label=f"true background / {key}")
            ax.set_yscale("log")
            ax.set_xlabel("Score")
            ax.set_ylabel("Relative frequency")
            ax.grid()
            ax.legend()
            fig.suptitle(f"Score distribution for {model_name} ensemble")
            scorefile = os.path.join(outdir, f"score_hist.{store_format}")
            fig.savefig(scorefile, dpi=250)
            if(config is not None):
                steganologger.encode(scorefile, dict(config=config, extra_info=extra_info), overwrite=True)
                if("score" in upload and not upload_fail):
                    resp = od_handler.upload(scorefile, f"R{config['RUN_ID']}_{config['TAG']}_{config['NAME']}-score-ens.{store_format}", to_path=f"Documenten/thesis/plots/{config['JOBNAME']}")
                    upload_fail = resp.status_code//100!=2
    if(len(upload)>0 and upload_fail):
        print("Upload failed")
if __name__=="__main__":
    import jlogger as jl
    import utils
    import argparse

    parser = argparse.ArgumentParser("Model trainer")
    parser.add_argument("--config", "-c", help="Path of the configuration file (.json or .py)", default='config.json', type=str)
    parser.add_argument("--outdir", "-o", help="Path of the output directory", type=str)
    parser.add_argument("--prediction", "-p", help="Path of the prediction files", nargs='*', default=['pred_final.h5', 'pred_best.h5'], type=str)
    parser.add_argument("--label", "-l", help="Labels of the prediction files", nargs='*', default=['final','best'], type=str)
    parser.add_argument("--plot_score", help="Plot score distribution", action='store_true')
    parser.add_argument("--logx", help="Use a logarithmic x-axis", action='store_true')
    parser.add_argument("--save", help="Save roc curves", choices=['roc', 'pred', 'mean', 'singles', 'all'], nargs='*', default='roc')
    parser.add_argument("--format", "-f", help="The file format", choices=["pdf", "svg", "png", "pgf"], default='png')
    parser.add_argument("--upload", "-u", help="Upload generated plots", nargs='*', type=str)
    args = vars(parser.parse_args())

    outdir = os.path.dirname(__file__)
    if(args["outdir"]):
        outdir = os.path.join(outdir, args["outdir"])
    print(f"Using output dir {outdir:s}")
    config_file = os.path.join(outdir, "0", args["config"])
    config = utils.configs.parse_config(config_file)
    print(f"Using config file {config_file:s}")

    predict_files = args["prediction"]
    assert len(args["label"])==len(args["prediction"]), "Arguments 'label' and 'prediction' should have same number of values"
    plot(outdir, model_name=config["MODELNAME"],
        prediction_files={args["label"][i]: os.path.join(outdir, "*", predict_files[i]) for i in range(min(len(predict_files), len(args["label"])))}, config=config,
        plot_score_hist=args["plot_score"], store_format=args['format'], save=args["save"], logx=args['logx'], upload=args['upload'] or [])