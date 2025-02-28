import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import steganologger, utils
import utils
import argparse
from scipy.optimize import curve_fit
import glob
from functools import reduce
from utils import shuffling

def plot(outdir:str, model_name:str="Model", prediction_files:dict=None,config:dict=None, fit_gaussian:bool=False, extra_info:dict=None,
         store_format:str='png', save:str=None):
    if(store_format=='pgf'):
        import matplotlib
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'font.size' : 11,
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

    def normal(x, A, mu, sigma):
        return A*1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(6,3))
    results = {}

    colors = utils.colors.ColorIter()
    target_color = next(colors)
    save_reco, save_mean = False, False
    if(save=='all' or 'reco' in save):
        savefile_reco = h5py.File(os.path.join(outdir, "test-reco.h5"), 'w')
        save_reco = True
    if(save=='all' or 'mean' in save):
        savefile_mean = h5py.File(os.path.join(outdir, "test-reco-mean.h5"), 'w')
        save_mean = True
    for i,key in enumerate(prediction_files.keys()):
        files = glob.glob(prediction_files[key], recursive=False)
        targets, recos = [], []
        if save_reco: savefile_reco.create_group(key)
        print(files)
        for file in files:
                h5pyfile = h5py.File(file, mode='r')
                idx = np.array(h5pyfile["data_idx"])
                reco, target = np.array(h5pyfile["reco"]), np.array(h5pyfile["target"])
                shuffling.do_the_unshuffle(reco, idx)
                shuffling.do_the_unshuffle(target, idx)
                targets.append(target)
                recos.append(reco)
                h5pyfile.close()
        assert all(np.allclose(targets[i],targets[0]) for i in range(1,len(targets))), "Not all targets match"
        stack = np.stack(recos)
        print(np.corrcoef(stack, rowvar=True))
        is_nan = reduce(lambda a,b: a | b, (np.isnan(reco) for reco in recos))
        print(f"Predicted {np.sum(is_nan)} nan's")
        target = targets[0]
        if save_reco:
            for j,reco in enumerate(recos):
                dataset = savefile_reco[key].create_dataset(f"{j}", data=reco)
                dataset.attrs["file"] = files[j]
            savefile_reco[key].create_dataset("targets", data=target)
 
        color = next(colors)
        reco = np.mean(stack, axis=0)
        if save_mean:
            dataset = savefile_mean.create_dataset(key, data=reco)
            dataset.attrs["files"] = files
            if i == 0:
                savefile_mean.create_dataset("target", data=target)
            else:
                assert np.all(np.array(savefile_mean["target"])==target), "Labels do not match"
        
        ax1.hist(reco/1000,   bins=np.linspace(2, 5, 100), histtype='step', linestyle='-', color=color, label=key)
        ax2.hist((target-reco)/1000, bins=np.linspace(-1,2.5,100), histtype='step', color=color, label=None)

        if(fit_gaussian):
            #try to fit a gaussian
            shift = np.mean(np.log10(target))
            x = np.log10(reco)-shift
            y, bins = np.histogram(x, bins=100)
            print(shift)
            print(bins)
            print(y)
            x = (bins[:-1]+bins[1:])/2
            popt, pcov = curve_fit(normal, x, y, p0=(1,0,1))
            print(popt)
            print(pcov)
            y = normal(x, *popt)
            x = 10**(x+shift)
            ax1.plot(x/1000, y, 'k--', label="Gaussian")

    #plot the target only once (we assume that all models have the same target)
    ax1.hist(target/1000, bins=np.linspace(2, 5, 100), histtype='step', color=target_color, label="Target")

    ax1.set_xlabel(r"$M_{JJ}$ [TeV]")
    ax1.set_ylabel("Freq.")
    ax2.set_xlabel(r"$M_{JJ}^{target}-M_{JJ}^{reco}$ [TeV]")
    ax2.set_ylabel("Freq.")
    ax1.set_xlim(2,5)
    ax2.set_xlim(-1,2.5)
    ax1.grid()
    ax2.grid()
    fig.legend()
    utils.formatting.format_floats(results, "{:.4f}".format)
    plotfile = os.path.join(outdir, f"ens_reco.{store_format}")
    fig.savefig(plotfile, dpi=250, bbox_inches='tight')
    steganologger.encode(plotfile, dict(results=results, extra_info=extra_info, config=config), overwrite=True)
    if(store_format=='pgf'):
        plotfile = os.path.join(outdir, f"ens_reco.png")
        fig.savefig(plotfile, dpi=250, bbox_inches='tight')
        steganologger.encode(plotfile, dict(results=results, config=config), overwrite=True)

if __name__=="__main__":
    import utils
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path of the configuration file (.json or .py)", default='config.json', type=str)
    parser.add_argument("--outdir", "-o", help="Path of the output directory", type=str)
    parser.add_argument("--prediction", "-p", help="Path of the prediction files", nargs='*', default=['pred_final.h5', 'pred_best.h5'], type=str)
    parser.add_argument("--label", "-l", help="Labels of the prediction files", nargs='*', default=['final','best'], type=str)
    parser.add_argument("--save", help="Save roc curves", choices=['roc', 'pred', 'mean', 'singles', 'all'], nargs='*', default='roc')
    parser.add_argument("--format", "-f", help="The file format", choices=["pdf", "svg", "png"], default='png')
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
        store_format=args['format'], save=args["save"])