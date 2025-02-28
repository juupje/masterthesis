import matplotlib.pyplot as plt
import h5py
import os, glob
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import steganologger, utils
from functools import reduce

def finish_roc_plot(fig, ax, logx=False, title=None):
    # random classifier
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
    fig.suptitle(title)

def finish_sic_plot(fig, ax, title=None):
    ax.set_xlabel(r"$\epsilon_S$")
    ax.set_ylabel(r"$\epsilon_S/\sqrt{\epsilon_B}$")
    ax.grid()
    ax.legend()
    fig.suptitle(title)

def plot_sic_curve(tpr, fpr, max_err, n_background, ax, inv_fpr=None, color=None, alpha=None, label=None):
    tpr_sic, sic = utils.sic.sic_curve(n_background, tpr, fpr=fpr, inv_fpr=inv_fpr, max_err=max_err)
    ax.plot(*utils.misc.thin_plot(tpr_sic, sic, 1000), color=color, alpha=alpha, label=label)
    return np.max(sic)

def plot_roc_curve(y_true, y_pred, ax, color=None, alpha=None, label=None, calc_stats=True):
    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    inv_fpr = 1/(fpr+1e-25)
    inv_fpr[fpr<1e-20] = np.nan
    ax.plot(*utils.misc.thin_plot(tpr, inv_fpr, 1000), color=color, alpha=alpha, label=label)
    if(calc_stats):
        result = {}
        for eps_s in (0.2,0.3,0.4,0.5):
            res = utils.calc.extract_at_value(tpr, eps_s, others=[fpr,thres], thres=0.01)
            if(res):  _, [fpr1, thres1] = res
            result[f"@eps_s={eps_s:.1f}"] = {"1/eps_b": 1/fpr1, "thres": thres1} if res else {}
        for eps_b in (1e-3,1e-4):
            res = utils.calc.extract_at_value(fpr, eps_b, others=[tpr,thres], thres=0.01)
            if(res):  _, [_tpr, _thres] = res
            result[f"@1/eps_b={1/eps_b:.1e}"] = {"eps_s": _tpr, "thres": _thres} if res else {}
        auroc = roc_auc_score(y_true, y_pred)
        result["auroc"] = auroc
        return (result, auroc), (fpr, tpr, thres)
    return fpr, tpr, thres

upload_failed = False
od_handler = None
def save_figure(fig, path:str, log_info:dict=None, upload_name:str|None=None, upload_path:str|None=None):
    global upload_failed, od_handler
    store_format = os.path.splitext(path)[1]
    fig.savefig(path, dpi=250)
    if(store_format=='pgf'):
        fig.savefig(path.replace(".pgf", ".png"), dpi=250)
    if log_info is not None:
        steganologger.encode(path, data=log_info, overwrite=True)
    if(upload_name is not None and not upload_fail):
        if od_handler is None:
            from onedrive import onedrive
            od_handler = onedrive.OneDriveHandler()
        #resp = od_handler.upload(path, f"R{config['RUN_ID']}_{config['TAG']}_{config['NAME']}-{name}-ens-fold{fold}.{store_format}", to_path=f"Documenten/thesis/plots/{config['JOBNAME']}")
        resp = od_handler.upload(path, upload_name, to_path=upload_path)
        upload_fail = resp.status_code//100!=2

def plot(outdir:str, logger=None, model_name:str="Model", prediction_files:dict=None,config:dict=None, extra_info:dict=None,
         plot_score_hist:bool=False, plot_sic:bool=False, store_format:str='png', save:list=None, logx:bool=False, upload:set=set()):
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
    if(prediction_files is None or len(prediction_files.keys())==0): return
        
    old = np.seterr(divide='ignore')
    if(save):
        savefile = h5py.File(os.path.join(outdir, "test-roc.h5"), 'w')

    #figure out the number of folds
    if "CROSS_VALIDATION" in config:
        nfolds = config["CROSS_VALIDATION"]["K"]
        assert len(config["CROSS_VALIDATION"]["test"])==1, "More than one test-split is not handled yet"
    else:
        raise ValueError("No cross-validation in config file.")

    # plot the roc curves
    fold_results = []
    fold_s_vs_b_pred = [] #this is performed on the same dataset for each fold
    fold_sr_vs_cr_pred = [] #this is the actual test-split of each fold
    for fold in range(nfolds):
        print(f"********** FOLD {fold} **********")
        if(save):
            fold_group = savefile.create_group(f"fold{fold}")
        fig, ax = plt.subplots()
        if(plot_sic):
            fig_sic, ax_sic = plt.subplots()
        colors = utils.colors.ColorIter()
        count = 0
        results = {}
        fold_s_vs_b_pred.append({})
        fold_sr_vs_cr_pred.append({})
        for i,key in enumerate(prediction_files.keys()):
            print(f" ==== {key} ==== ")
            if(save):
                key_group = fold_group.create_group(key)
                key_group.create_group("singles")
            results[key] = {}
            #collect all the prediction files
            print(os.path.join(outdir, '*', f"fold{fold}", prediction_files[key]))
            files = glob.glob(os.path.join(outdir, '*', f"fold{fold}", prediction_files[key]), recursive=False)
            print(files)
            labels, y_preds = [], []
            cr = {"labels": [], "y_preds": []}
            sr = {"labels": [], "y_preds": []}
            plot_cr_vs_sr = True
            for file in files: #Find all the repeats
                h5pyfile = h5py.File(file, mode='r')
                idx = np.array(h5pyfile["s_vs_b/data_idx"])
                y_pred, y_true = np.array(h5pyfile["s_vs_b/pred"][:,1]), np.array(h5pyfile["s_vs_b/label"])
                y_pred, y_true = utils.misc.restore_ordering(y_pred, y_true, idx)
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
            
            # ====== PLOT SIGNAL VS BACKGROUND ====== 
            assert all(np.all(labels[i]==labels[0]) for i in range(1,len(labels))), "Not all labels match"
            stack = np.stack(y_preds)
            print(np.corrcoef(stack, rowvar=True))
            is_nan = reduce(lambda a,b: a | b, (np.isnan(y_pred) for y_pred in y_preds))
            print(f"Predicted {np.sum(is_nan)} nan's")
            y_true = labels[0][~is_nan]
            n_background = np.sum(y_true==0)
            y_pred = np.mean(stack[:,~is_nan], axis=0)
            fold_s_vs_b_pred[fold][key] = {"y_pred": y_pred, "y_true": y_true, "ensemble_size": len(y_preds)}
            
            color = next(colors)
            (res, auroc), (fpr, tpr, thres) = plot_roc_curve(y_true, y_pred, ax=ax, color=color, label=key, calc_stats=True)
            ax.text(0.4,0.8-count*0.05, f"AUROC={auroc:.4f}", color=color, transform=ax.transAxes)
            results[key] = res
            if(save):
                key_group.create_dataset("mean", data=np.stack((fpr,tpr,thres),axis=1))
                
            if(plot_sic):
                max_sic = plot_sic_curve(tpr, fpr, max_err=0.2, n_background=n_background, ax=ax_sic, color=color, label=key)
                ax_sic.text(0.4,0.8-count*0.05, f"MAX={max_sic:.4f}", color=color, transform=ax_sic.transAxes)
            count += 1

            #plot the individual rocs
            for i, x in enumerate(y_preds):
                fpr, tpr, thres = plot_roc_curve(y_true, x, ax, color=color, alpha=0.3, calc_stats=False)
                if(plot_sic):
                    plot_sic_curve(tpr, fpr, max_err=0.2, n_background=n_background, ax=ax_sic, color=color, alpha=0.3)
                if(save):
                    key_group["singles"].create_dataset(f"roc{i}", data=np.stack((fpr,tpr,thres),axis=1))

            # ====== EVALUATE SR VS SB ====== 
            if(plot_cr_vs_sr):
                assert all(np.all(sr["labels"][i]==sr["labels"][0]) for i in range(1,len(sr["labels"]))), "Not all labels match (sr)"
                assert all(np.all(cr["labels"][i]==cr["labels"][0]) for i in range(1,len(cr["labels"]))), "Not all labels match (cr)"
                cr_stack = np.stack(cr["y_preds"])
                sr_stack = np.stack(sr["y_preds"])
                y_pred_cr = np.mean(cr_stack, axis=0)
                y_true_cr = cr["labels"][0]
                y_pred_sr = np.mean(sr_stack, axis=0)
                y_true_sr = sr["labels"][0]
                fold_sr_vs_cr_pred[fold][key] = {"y_pred": np.concatenate((y_pred_cr, y_pred_sr),axis=0), 
                                                 "y_true": np.concatenate((y_true_cr, y_true_sr),axis=0,dtype=np.int8),
                                                 "y_weak": np.concatenate((np.zeros_like(y_pred_cr,dtype=np.int8), np.ones_like(y_pred_sr, dtype=np.int8)),axis=0),
                                                 "ensemble_size": len(cr["y_preds"])}
                results[key]["test"] = utils.evaluation.evaluate_sr_vs_cr(y_pred_cr=y_pred_cr, y_true_cr=y_true_cr, y_pred_sr=y_pred_sr, y_true_sr=y_true_sr, n=1000)
        
        #Finish up the plots
        utils.format_floats(results, "{:.4f}".format)
        fold_results.append(results)
        finish_roc_plot(fig, ax, logx=logx, title=f"ROC curve for {model_name} ensemble / fold {fold}")
        rocfile = os.path.join(outdir,f"roc_fold{fold}.{store_format}")
        log_info = dict(config=config, results=fold_results, extra_info=extra_info)
        save_figure(fig, rocfile, log_info=log_info,
                    upload_name = f"R{config['RUN_ID']}_{config['TAG']}_{config['NAME']}-roc-ens-fold{fold}.{store_format}" if {"roc","folds"}.issubset(upload) else None,
                    upload_path = f"Documenten/thesis/plots/{config['JOBNAME']}")
        if(plot_sic):
            finish_sic_plot(fig_sic, ax_sic, title=f"SIC curve for {model_name} ensemble / fold {fold}")
            sicfile = os.path.join(outdir,f"sic_fold{fold}.{store_format}")
            save_figure(fig_sic, sicfile, log_info=log_info,
                        upload_name = f"R{config['RUN_ID']}_{config['TAG']}_{config['NAME']}-sic-ens-fold{fold}.{store_format}" if {"roc", "folds"}.issubset(upload) else None,
                        upload_path = f"Documenten/thesis/plots/{config['JOBNAME']}")

        #Plot the score distribution
        if(plot_score_hist):
            fig, ax = plt.subplots()
            bins = {}
            colors = utils.colors.ColorIter()
            for i,key in enumerate(prediction_files):
                files = glob.glob(os.path.join(outdir, '*', f"fold{fold}", prediction_files[key]), recursive=False)
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
            fig.suptitle(f"Score distribution for {model_name} ensemble / fold {fold}")
            scorefile = os.path.join(outdir, f"score_hist_fold{fold}.{store_format}")
            save_figure(fig, scorefile, dict(config=config, extra_info=extra_info) if config else None,
                        upload_name=f"R{config['RUN_ID']}_{config['TAG']}_{config['NAME']}-score-ens_fold{fold}.{store_format}" if {"score", "folds"}.issubset(upload) else None,
                        upload_path=f"Documenten/thesis/plots/{config['JOBNAME']}")

    #Now we can plot the combined result of all the folds
    # and evaluate the complete test set

    print(" ************ COMBINED **********")
    #1. stack all s-vs-b test set predictions
    coloriter = utils.colors.ColorIter()
    fig, ax = plt.subplots()
    if(plot_sic):
        fig_sic, ax_sic = plt.subplots()
    count = 0
    s_vs_b_results = {}
    sr_vs_cr_results = {}
    combined_file = h5py.File(os.path.join(outdir, "combined_pred.h5"), 'w')
    s_vs_b_group = combined_file.create_group("s_vs_b")
    sr_vs_cr_group = combined_file.create_group("sr_vs_cr")
    for label in prediction_files.keys():
        print(f" ==== {label} =====")
        total_svsb_trues = np.stack([fold_s_vs_b_pred[i][label]["y_true"] for i in range(nfolds)], axis=0)
        #2. assert that all labels are the same
        assert np.all(total_svsb_trues[1:]==total_svsb_trues[0]), "signal vs. background labels do not match across folds"
        total_svsb_true = total_svsb_trues[0]
        #3. take the mean of all folds (weighted by the number of repeats per fold)
        weights = np.array([fold_s_vs_b_pred[i][label]["ensemble_size"] for i in range(nfolds)])
        total_svsb_pred = np.average(np.stack([fold_s_vs_b_pred[i][label]["y_pred"] for i in range(nfolds)], axis=0),weights=weights, axis=0)
        #save the predictions
        group = s_vs_b_group.create_group(label)
        group.create_dataset("pred", data=total_svsb_pred)
        group.create_dataset("label", data=total_svsb_true)

        #4. plot the resulting roc curve
        color = next(coloriter)
        (res, auroc), (fpr, tpr, thres) = plot_roc_curve(total_svsb_true, total_svsb_pred, ax, color=color, calc_stats=True, label=label)
        s_vs_b_results[label] = res
        ax.text(0.4,0.8-count*0.05, f"AUROC={auroc:.4f}", color=color, transform=ax.transAxes)
        if(save):
            #save the roc curves
            group = savefile.create_group(label)
            group.create_dataset("mean", data=np.stack((tpr, fpr, thres),axis=1))
        
        if(plot_sic):
            n_background = np.sum(total_svsb_true==0)
            plot_sic_curve(tpr, fpr, max_err=0.2, n_background=n_background, ax=ax_sic, color=color, label=label)
        for i in range(nfolds):
            fpr, tpr, _ = plot_roc_curve(total_svsb_trues[i], fold_s_vs_b_pred[i][label]["y_pred"], ax, color=color, alpha=0.3, calc_stats=False)
            if(plot_sic):
                plot_sic_curve(tpr, fpr, max_err=0.2, n_background=n_background, ax=ax_sic, color=color, alpha=0.3)

    
        #5. concatenate the sr-vs-cr test split predictions of the folds
        y_true = np.concatenate([fold_sr_vs_cr_pred[fold][label]["y_true"] for fold in range(nfolds)], axis=0)
        y_true_weak = np.concatenate([fold_sr_vs_cr_pred[fold][label]["y_weak"] for fold in range(nfolds)], axis=0)
        y_pred = np.concatenate([fold_sr_vs_cr_pred[fold][label]["y_pred"] for fold in range(nfolds)], axis=0)
        group = sr_vs_cr_group.create_group(label)
        group.create_dataset("pred", data=y_pred)
        group.create_dataset("label", data=y_true)
        group.create_dataset("weak_label", data=y_true_weak)
    
        #6. assert that the number of repeats per fold is the same
        #todo

        #7. compute the stats over the entire test 
        is_cr = y_true_weak==0
        y_pred_cr, y_true_cr = y_pred[is_cr], y_true[is_cr]
        y_pred_sr, y_true_sr = y_pred[~is_cr], y_true[~is_cr]
        sr_vs_cr_results[label] = utils.evaluation.evaluate_sr_vs_cr(y_pred_cr=y_pred_cr, y_true_cr=y_true_cr, y_pred_sr=y_pred_sr, y_true_sr=y_true_sr, n=1000)

    utils.format_floats(sr_vs_cr_results, "{:.4f}".format)
    utils.format_floats(s_vs_b_results, "{:.4f}".format)
    with open(os.path.join(outdir, "eval_result.json"), 'w') as f:
        import json
        json.dump(sr_vs_cr_results, f, indent=2, cls=utils.formatting.Encoder)

    finish_roc_plot(fig, ax, logx=logx, title=f"Signal vs. background ensemble of {nfolds} folds")
    rocfile = os.path.join(outdir,f"roc_fold_combined.{store_format}")
    log_info = dict(config=config, s_vs_b=s_vs_b_results, sr_vs_cr=sr_vs_cr_results, extra_info=extra_info)
    save_figure(fig, rocfile, log_info=log_info,
                upload_name=f"R{config['RUN_ID']}_{config['TAG']}_{config['NAME']}-roc-total.{store_format}" if "roc" in upload else None,
                upload_path=f"Documenten/thesis/plots/{config['JOBNAME']}")
    
    if(plot_sic):
        finish_sic_plot(fig_sic, ax_sic, title=f"Signal vs. background ensemble of {nfolds} folds")
        sicfile = rocfile.replace("roc", "sic")
        save_figure(fig_sic, sicfile, log_info=log_info,
                    upload_name=f"R{config['RUN_ID']}_{config['TAG']}_{config['NAME']}-sic-total.{store_format}" if "sic" in upload else None,
                    upload_path=f"Documenten/thesis/plots/{config['JOBNAME']}")
    
    if(save):
        savefile.close()
    combined_file.close()
    np.seterr(**old)
    
    if(len(upload)>0 and upload_fail):
        print("Upload failed")
if __name__=="__main__":
    import jlogger as jl
    import utils
    import argparse

    parser = argparse.ArgumentParser("Model trainer")
    parser.add_argument("--config", "-c", help="Path of the configuration file (.json or .py)", default='config.json', type=str)
    parser.add_argument("--outdir", "-o", help="Path of the output directory", type=str)
    #parser.add_argument("--prediction", "-p", help="Path of the prediction files", nargs='*', type=str)
    parser.add_argument("--label", "-l", help="Labels of the prediction files", nargs='*', default=['final','best'], type=str)
    parser.add_argument("--plot_score", help="Plot score distribution", action='store_true')
    parser.add_argument("--logx", help="Use a logarithmic x-axis", action='store_true')
    parser.add_argument("--sic", help="Plot SIC curves", action='store_true')
    parser.add_argument("--save", help="Save ROC curves", action='store_true')
    parser.add_argument("--format", "-f", help="The file format", choices=["pdf", "svg", "png", "pgf"], default='png')
    parser.add_argument("--upload", "-u", help="Upload generated plots", nargs='*', type=str)
    args = vars(parser.parse_args())

    outdir = os.path.dirname(__file__)
    if(args["outdir"]):
        outdir = os.path.join(outdir, args["outdir"])
    print(f"Using output dir {outdir:s}")
    config_file = os.path.join(outdir, "0", args["config"])
    if not os.path.isfile(config_file):
        config_file = os.path.join(outdir, args["config"])
    config = utils.parse_config(config_file)
    print(f"Using config file {config_file:s}")

    #if args["prediction"] is None:
    predict_files = [f"pred_{label}.h5" for label in args["label"]]
    #else:
    #    predict_files = args["prediction"]
    #    assert len(args["label"])==len(args["prediction"]), "Arguments 'label' and 'prediction' should have same number of values"
    plot(outdir, jl.JLogger(), model_name=config["MODELNAME"],
        prediction_files={args["label"][i]: predict_files[i] for i in range(min(len(predict_files), len(args["label"])))}, config=config,
        plot_score_hist=args["plot_score"], store_format=args['format'], plot_sic=args["sic"], save=args["save"], logx=args['logx'], upload=set(args['upload'] or []))