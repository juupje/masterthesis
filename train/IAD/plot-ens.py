import matplotlib.pyplot as plt
import h5py
import os, glob
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import steganologger, utils
from functools import reduce
from utils import shuffling
def plot(outdir:str, logger, model_name:str="Model", prediction_files:dict=None,config:dict=None, extra_info:dict=None,
         plot_score_hist:bool=False, plot_sic:bool=False, store_format:str='png', save:list=None, logx:bool=False, upload:list=[]):
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
        #plot the ROC curve
        fig, ax = plt.subplots()
        results = {}
        colors = utils.colors.ColorIter()
        count = 0
        if(save):
            savefile = h5py.File(os.path.join(outdir, "test-roc.h5"), 'w')

        if(plot_sic):
            fig_sic, ax_sic = plt.subplots()

        for i,key in enumerate(prediction_files.keys()):
            #collect all the prediction files
            files = glob.glob(prediction_files[key], recursive=False)
            labels, y_preds = [], []
            cr = {"labels": [], "y_preds": []}
            sr = {"labels": [], "y_preds": []}
            plot_cr_vs_sr = True
            print(files)
            for file in files:
                h5pyfile = h5py.File(file, mode='r')
                idx = np.array(h5pyfile["s_vs_b/data_idx"])
                y_pred, y_true = np.array(h5pyfile["s_vs_b/pred"][:,1]), np.array(h5pyfile["s_vs_b/label"])
                y_pred, y_true = utils.misc.restore_ordering(y_pred, y_true, idx)
                print(y_true)
                #y_pred, y_true = y_pred[sorted_idx], y_true[sorted_idx]
                labels.append(y_true)
                y_preds.append(y_pred)

                if("cr" in h5pyfile and "sr" in h5pyfile):
                    y_pred, y_true = np.array(h5pyfile["cr/pred"][:,1]), np.array(h5pyfile["cr/label"])
                    cr["labels"].append(y_true)
                    cr["y_preds"].append(y_pred)
                    y_pred, y_true = np.array(h5pyfile["sr/pred"][:,1]), np.array(h5pyfile["sr/label"])
                    sr["labels"].append(y_true)
                    sr["y_preds"].append(y_pred)
                else:
                    plot_cr_vs_sr = False
                    print(f"File {file} does not contain cr or sr predictions")
                h5pyfile.close()
            assert all(np.all(labels[i]==labels[0]) for i in range(1,len(labels))), "Not all labels match"
            stack = np.stack(y_preds)
            print(np.corrcoef(stack, rowvar=True))
            is_nan = reduce(lambda a,b: a | b, (np.isnan(y_pred) for y_pred in y_preds))
            print(f"Predicted {np.sum(is_nan)} nan's")
            y_true = labels[0][~is_nan]
            n_background = np.sum(y_true==0)

            if(plot_cr_vs_sr):
                assert all(np.all(sr["labels"][i]==sr["labels"][0]) for i in range(1,len(sr["labels"]))), "Not all labels match (sr)"
                assert all(np.all(cr["labels"][i]==cr["labels"][0]) for i in range(1,len(cr["labels"]))), "Not all labels match (cr)"
                cr_stack = np.stack(cr["y_preds"])
                sr_stack = np.stack(sr["y_preds"])
            
            if(save):
                key_group = savefile.create_group(key)
                key_group.create_group("singles")
            #Just try a couple of different averaging functions
            results[key] = {}
            color = next(colors)
            for func,name,brightness in zip((np.mean,np.median,np.max), ("mean", "median", "max"), (0.2,0.4,0.6)):
                y_pred = func(stack, axis=0)
                results[key][name] = {}
                fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
                if(save and name in save):
                    savefile[key].create_dataset(name, data=np.stack((fpr,tpr,thres),axis=1))

                for eps_s in (0.2,0.3,0.4,0.5):
                    res = utils.calc.extract_at_value(tpr, eps_s, others=[fpr,thres], thres=0.01)
                    if(res):  _, [fpr1, thres1] = res
                    results[key][name][f"@eps_s={eps_s:.1f}"] = {"1/eps_b": 1/fpr1, "thres": thres1} if res else {}
                for eps_b in (1e-3,1e-4):
                    res = utils.calc.extract_at_value(fpr, eps_b, others=[tpr,thres], thres=0.01)
                    if(res):  _, [_tpr, _thres] = res
                    results[key][name][f"@1/eps_b={1/eps_b:.1e}"] = {"eps_s": _tpr, "thres": _thres} if res else {}
                auroc = roc_auc_score(y_true, y_pred)
                results[key][name]["auroc"] = auroc

                inv_fpr = 1/fpr
                inv_fpr[np.isinf(inv_fpr)] = np.nan
                c = utils.colors.change_brightness(color, brightness)
                ax.plot(*utils.misc.thin_plot(tpr, inv_fpr, 1000), color=c,label=f"{key} - {name}")
                ax.text(0.4,0.8-count*0.05, f"AUROC={auroc:.4f}", color=c, transform=ax.transAxes)

                if(plot_sic):
                    max_err = 0.2
                    tpr_sic, sic = utils.sic.sic_curve(n_background, tpr, inv_fpr=inv_fpr, max_err=max_err)
                    ax_sic.plot(*utils.misc.thin_plot(tpr_sic, sic, 1000), color=c, label=f"{key} - {name}")
                    ax_sic.text(0.4,0.8-count*0.05, f"MAX={np.max(sic):.4f}", color=c, transform=ax_sic.transAxes)
                count += 1

                if(plot_cr_vs_sr):
                    y_pred_cr = func(cr_stack, axis=0)
                    y_true_cr = cr["labels"][0]
                    y_pred_sr = func(sr_stack, axis=0)
                    y_true_sr = sr["labels"][0]
                    to_select = y_pred_cr.shape[0]//1000
                    thres = np.partition(y_pred_cr, -to_select)[-to_select]
                
                    print(f"===== {name} =====")
                    print("At eps_b=0.1%, selected:")
                    print("\tIn CR:")
                    is_selected_cr = y_pred_cr>=thres
                    is_background_cr = y_true_cr==0
                    n_background_in_cr = np.sum(is_selected_cr&is_background_cr)
                    n_signal_in_cr = np.sum(is_selected_cr& ~is_background_cr)
                    print(f"\t\t{n_background_in_cr:d} background events of {np.sum(is_background_cr):d}")
                    print(f"\t\t{n_signal_in_cr:d} signal events of {np.sum(~is_background_cr):d}")
                    print("\tIn SR:")
                    is_selected_sr = y_pred_sr>=thres
                    is_background_sr = y_true_sr==0
                    n_background_in_sr = np.sum(is_selected_sr&is_background_sr)
                    n_signal_in_sr = np.sum(is_selected_sr& ~is_background_sr)
                    print(f"\t\t{np.sum(is_selected_sr):d} events selected in total")
                    print(f"\t\t{n_background_in_sr:d} background events of {np.sum(is_background_sr):d}")
                    print(f"\t\t{n_signal_in_sr:d} signal events of {np.sum(~is_background_sr):d}")
                    expected_background_in_sr = n_background_in_cr*(y_pred_sr.shape[0]/y_pred_cr.shape[0])
                    sigma = np.sqrt(n_background_in_cr+expected_background_in_sr)
                    significance = (np.sum(is_selected_sr)-expected_background_in_sr)/sigma
                    print(f"\t\tExpected events in SR: {expected_background_in_sr}")
                    print(f"\tSigma: {sigma:.2f}, significance: {significance:.2f}")

                    results[key][name]["test"] = {"eps_b": 0.001,
                                                "cr": {"background": np.sum(is_background_cr), "signal": np.sum(~is_background_cr),
                                                        "selected": {"total": np.sum(is_selected_cr), "background": np.sum(n_background_in_cr), "signal": np.sum(n_signal_in_cr)}},
                                                "sr": {"background": np.sum(is_background_sr), "signal": np.sum(~is_background_sr),
                                                        "selected": {"total": np.sum(is_selected_sr), "background": np.sum(n_background_in_sr), "signal": np.sum(n_signal_in_sr)}},
                                                "sigma": sigma,
                                                "significance": significance}

            for i, x in enumerate(y_preds):
                fpr, tpr, thres = roc_curve(y_true, x, pos_label=1)
                inv_fpr = 1/fpr
                inv_fpr[np.isinf(inv_fpr)] = np.nan
                ax.plot(*utils.misc.thin_plot(tpr, inv_fpr, 1000), color=color, alpha=0.3)
                if(plot_sic):
                    ax_sic.plot(*utils.misc.thin_plot(*utils.sic.sic_curve(n_background, tpr, inv_fpr=inv_fpr), 1000), color=color, alpha=0.3)
                if(save):
                    savefile[key]["singles"].create_dataset(f"roc{i}", data=np.stack((fpr,tpr,thres),axis=1))

        if(save):
            savefile.close()
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

        utils.format_floats(results, "{:.4f}".format)
        rocfile = os.path.join(outdir,f"roc.{store_format}")
        sicfile = os.path.join(outdir,f"sic.{store_format}")
        logger.log_figure(fig, rocfile, data_file=", ".join(list(prediction_files.values())), dpi=250)
        if(plot_sic):
            ax_sic.set_xlabel(r"$\epsilon_S$")
            ax_sic.set_ylabel(r"$\epsilon_S/\sqrt{\epsilon_B}$")
            ax_sic.grid()
            ax_sic.legend()
            fig_sic.suptitle(f"SIC curve for {model_name} ensemble")
            logger.log_figure(fig_sic, sicfile, data_file=", ".join(list(prediction_files.values())), dpi=250)
        for file,name in zip([rocfile, sicfile], ["roc", "sic"]) if plot_sic else zip([rocfile], ["roc"]):
            if(config is not None):
                steganologger.encode(file, dict(config=config, results=results, extra_info=extra_info), overwrite=True)
                if("roc" in upload and not upload_fail):
                    resp = od_handler.upload(file, f"R{config['RUN_ID']}_{config['TAG']}_{config['NAME']}-{name}-ens.{store_format}", to_path=f"Documenten/thesis/plots/{config['JOBNAME']}")
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
                    idx = np.array(h5pyfile["s_vs_b/data_idx"])
                    y_pred, y_true = np.array(h5pyfile["s_vs_b/pred"][:,1]), np.array(h5pyfile["s_vs_b/label"])
                    y_pred, y_true = utils.misc.restore_ordering(y_pred, y_true, idx)
                    labels.append(y_true)
                    y_preds.append(y_pred)
                    h5pyfile.close()
                assert all(np.all(labels[i]==labels[0]) for i in range(1,len(labels))), "Not all labels match"
                is_nan = reduce(lambda a,b: a | b, (np.isnan(y_pred) for y_pred in y_preds))
                y_pred = np.mean(np.stack(y_preds), axis=0)
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
            logger.log_figure(fig, scorefile, data_file=", ".join(list(prediction_files.values())), dpi=250)
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
    parser.add_argument("--prediction", "-p", help="Path of the prediction files", nargs='*', type=str)
    parser.add_argument("--label", "-l", help="Labels of the prediction files", nargs='*', default=['final','best'], type=str)
    parser.add_argument("--plot_score", help="Plot score distribution", action='store_true')
    parser.add_argument("--logx", help="Use a logarithmic x-axis", action='store_true')
    parser.add_argument("--sic", help="Plot SIC curves", action='store_true')
    parser.add_argument("--save", help="Save ROC/SIC curves", nargs='*')
    parser.add_argument("--format", "-f", help="The file format", choices=["pdf", "svg", "png", "pgf"], default='png')
    parser.add_argument("--upload", "-u", help="Upload generated plots", nargs='*', type=str)
    args = vars(parser.parse_args())

    outdir = os.path.dirname(__file__)
    if(args["outdir"]):
        outdir = os.path.join(outdir, args["outdir"])
    print(f"Using output dir {outdir:s}")
    config_file = os.path.join(outdir, "0", args["config"])
    config = utils.parse_config(config_file)
    print(f"Using config file {config_file:s}")

    if args["prediction"] is None:
        predict_files = [f"pred_{label}.h5" for label in args["label"]]
    else:
        predict_files = args["prediction"]
        assert len(args["label"])==len(args["prediction"]), "Arguments 'label' and 'prediction' should have same number of values"
    plot(outdir, jl.JLogger(), model_name=config["MODELNAME"],
        prediction_files={args["label"][i]: os.path.join(outdir, "*", predict_files[i]) for i in range(min(len(predict_files), len(args["label"])))}, config=config,
        plot_score_hist=args["plot_score"], store_format=args['format'], plot_sic=args["sic"], save=args["save"], logx=args['logx'], upload=args['upload'] or [])