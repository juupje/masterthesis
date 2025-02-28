import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
import utils

def list_datasets(file:h5py.Group):
    datasets = []
    def visit(name, obj):
        if(isinstance(obj, h5py.Dataset)):
            datasets.append(name)
    file.visititems(visit)
    return datasets

def plot(outfile:str, logger, files:list, labels:list, keys:list=None, logx:bool=False, ensemble_only:bool=False, store_format:str='png', upload:bool=False):
    old = np.seterr(divide='ignore')
    #plot the ROC curve
    fig, ax = plt.subplots()
    coloriter = utils.colors.ColorIter()
    for idx, f in enumerate(files):
        file = h5py.File(f, 'r')
        k = [keys[idx]] if keys else list_datasets(file)
        l = [labels[idx]] if labels else ("/".join(f.split("/")[:-1]) +" - " + key)
        c = utils.colors.change_brightness(next(coloriter),0.4)
        flag = False
        for key,label in zip(k,l):
            print(f, key, l)
            if(key.startswith("roc") and ensemble_only): continue
            data = np.array(file[key])
            print(data.shape)
            fpr, tpr = data[:,0], data[:,1]
            color = next(coloriter)
            inv_fpr = 1/fpr
            inv_fpr[np.isinf(inv_fpr)] = np.nan
            if(key.startswith("roc")):
                ax.plot(*utils.misc.thin_plot(tpr, inv_fpr,1000), color=c,label=None)
                flag = True
            else:
                ax.plot(*utils.misc.thin_plot(tpr, inv_fpr,1000), color=color,label=label)

    # random classifier
    np.seterr(**old)
    x = np.linspace(0.001,1,100)
    ax.plot(x, 1/x, 'k--')

    #Make it pretty
    ax.set_yscale('log')
    if(logx):
        ax.set_xscale('log')
        ax.set_xlim(1e-3,None)
    ax.set_xlim(0,1)
    ax.set_ylim(1,1e5)
    ax.set_xlabel(r"$\epsilon_S$")
    ax.set_ylabel(r"$1/\epsilon_B$")
    ax.grid()
    ax.legend()
    fig.suptitle(f"ROC curve comparison")
    if(store_format):
        outfile = f"{outfile}.{store_format}"
    logger.log_figure(fig, outfile, data_file=", ".join(files), dpi=250)
  
if __name__=="__main__":
    import jlogger as jl
    import utils
    import argparse

    parser = argparse.ArgumentParser("Model trainer")
    parser.add_argument("files", help="h5py files containing the rocs to be plotted", nargs='*', type=str)
    parser.add_argument("--key", "-k", help="keys of the h5py files to be plotted", nargs='*', type=str)
    parser.add_argument("--label", "-l", help="labels of the h5py files to be plotted", nargs='*', type=str)
    parser.add_argument("--output", "-o", help="Path of the prediction files", required=True, type=str)
    parser.add_argument("--logx", help="Logarithmic x-axis", action='store_true')
    parser.add_argument("--only_ensemble", "-e", help="Plot only the ensemble", action='store_true')
    parser.add_argument("--format", "-f", help="The file format", choices=["pdf", "svg", "png", "pgf"], default=None)
    parser.add_argument("--upload", "-u", help="Upload generated plot", action='store_true')
    args = vars(parser.parse_args())

    if args["key"]:
        if(len(args["key"])==1):
            args["key"] = args["key"]*len(args["files"])
        else:
            assert(len(args["key"])==len(args["files"])), "There should be a single key or one for every file"
    if args["label"]:
        # create labels from files
        if len(args["label"])==1:
            args["label"] = args["label"]*len(args["files"])
        else:
            assert(len(args["label"])==len(args["files"])), "There should be a single label or one for every file"
    plot(args["output"], jl.JLogger(), args["files"], keys=args["key"], labels=args["label"], logx=args['logx'], ensemble_only=args["only_ensemble"], store_format=args['format'], upload=args["upload"])