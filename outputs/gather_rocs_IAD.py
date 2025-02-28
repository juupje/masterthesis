import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
import utils
import argparse
import steganologger

parser = argparse.ArgumentParser()
parser.add_argument("--logx", help="Logarithmic x-axis", action='store_true')
parser.add_argument("--outfile", "-o", help="Name of the output file (without extension)", type=str, default="rocs")
parser.add_argument("--ensemble-only", "-e", help="Plot only the ensemble", action='store_true')
parser.add_argument("--format", "-f", help="The file format", choices=["pdf", "svg", "png", "pgf"], default="png")
parser.add_argument("--upload", "-u", help="Upload generated plot", action='store_true')
parser.add_argument("--signal", "-s", help="number of signal events", type=int, default=395)
args = vars(parser.parse_args())

old = np.seterr(divide='ignore')

if(args["format"] == "pgf"):
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size' : 11,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
colors = utils.colors.Colors()
rocs = {"LorentzNet": {
            "file": "IAD3_cathode_LN/" + ("R266_base_case" if args['signal']==395 else (f"R258g_more_signal/S={args['signal']:d}" if args["signal"]>395 else f"R177g_less_signal/S={args['signal']:d}"))+ "/test-roc.h5" ,
            "color": colors[0],
            "label": "LorentzNet\\textsubscript{$p_T$}"
        },
        "ParticleNet": {
            "label": "ParticleNet",
            "color": colors[1],
            "file": f"IAD2_cathode_PN/R264g_correct_phi_feature_test/S={args['signal']:d}/test-roc.h5"
        },
        "PELICAN": {
             "label": "PELICAN",
             "color": colors[2],
             "file": f"IAD2_cathode_PL/R288g_Scan/S={args['signal']:d}/test-roc.h5"
        }
        }


#plot the ROC curve
fig, ax = plt.subplots(figsize=(4.5,4))
coloriter = utils.colors.ColorIter()
results = {}
for idx, roc in enumerate(rocs.keys()):
    print(roc)
    file = h5py.File(rocs[roc]["file"], 'r')
    color = rocs[roc]["color"]
    color2 = utils.colors.change_brightness(color, 0.8)
    results[roc] = {}
    print(file["final"].keys())
    for key in file["final"]: # should only be 'mean'
        if type(file["final"][key]) is not h5py.Dataset: continue
        data = np.array(file['final'][key])
        print(rocs[roc]["label"], key)
        fpr, tpr = data[:,0], data[:,1]
        inv_fpr = 1/fpr
        inv_fpr[np.isinf(inv_fpr)] = np.nan
        if key not in results[roc]: results[roc][key] = {}
        for eps_s in (0.2,0.4,0.6):
            _,res = utils.calc.extract_at_value(tpr, eps_s, others=fpr, thres=0.01)
            results[roc][key][f"@eps_s={eps_s:.1f}"] = {"1/eps_b": 1/res} if res else {}
        for eps_b in (1e-3,1e-4):
            _,res = utils.calc.extract_at_value(fpr, eps_b, others=tpr, thres=0.01)
            results[roc][key][f"@1/eps_b={1/eps_b:.1e}"] = {"eps_s": res} if res else {}
        ax.plot(*utils.misc.thin_plot(tpr, inv_fpr,1000), color=color,label=rocs[roc]["label"])
    
    if not args["ensemble_only"]:
        if("singles" not in file["final"]):
            print("No single roc curves!")
        else:
            singles = file["final"]["singles"]
            for key in singles:
                data = np.array(singles[key])
                fpr, tpr = data[:,0], data[:,1]
                inv_fpr = 1/fpr
                inv_fpr[np.isinf(inv_fpr)] = np.nan
                ax.plot(*utils.misc.thin_plot(tpr, inv_fpr,1000), color=color2,label=None,alpha=0.3)
# random classifier
np.seterr(**old)
x = np.linspace(0.001,1,100)
ax.plot(x, 1/x, 'k--')

#Make it pretty
ax.set_yscale('log')
if(args["logx"]):
    ax.set_xscale('log')
    ax.set_xlim(1e-3,None)
ax.set_xlim(0,1)
ax.set_ylim(1,1e5)
ax.set_xlabel(r"$\epsilon_S$")
ax.set_ylabel(r"$1/\epsilon_B$")
ax.grid()
ax.legend()
#fig.suptitle(f"ROC curve comparison")
if(args["format"]):
    if(args["format"]!="png"):
        fig.savefig(os.path.join("IAD_result",args["outfile"]+".png"), dpi=250, bbox_inches='tight')
    outfile = args["outfile"] + "." + args['format']
fig.savefig(os.path.join("IAD_result",outfile), dpi=250, bbox_inches='tight')
steganologger.encode(os.path.join("IAD_result",outfile), data={"results":results, 'rocs': rocs}, overwrite=True)
if args["upload"]:
    from onedrive import onedrive
    od_handler = onedrive.OneDriveHandler()
    od_handler.upload(os.path.join("IAD_result",outfile), name=outfile, to_path="Documenten/thesis/plots/IAD")
    