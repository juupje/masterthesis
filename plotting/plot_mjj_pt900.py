#! /usr/env/python3
#Copy of plot_mjj.py, but for self-generated data
import os, sys
sys.path.append(os.getenv("HOME")+"/Analysis/lib")
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import utils
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

DATA_DIR = "/scratch/work/geuskens/data/lhco/new/"
PLOTS_DIR = os.getenv("HOME") + "/Analysis/plots/data_gen"

SR = (3300,3700)
SB_LEFT = 2900
SB_RIGHT = 4100

N = 1000000
factor = 1
file_bg = h5py.File(os.path.join(DATA_DIR, "N100-bg-combined.h5"),'r')
file_sn = h5py.File(os.path.join(DATA_DIR, "pt900/N100-sn.h5"),'r')
features_bg = np.array(file_bg["jet_features"][:N,-1])
n_s = int(1e-1*features_bg.shape[0])
features_sg = np.array(file_sn["jet_features"][:n_s*factor,-1])

fig = plt.figure(figsize=(8,7))
gs = GridSpec(2,4)
ax = fig.add_subplot(gs[0,:])
bins_mjj = 200
coloriter = utils.colors.ColorIter()
colors = (next(coloriter),next(coloriter))
mjjs = {"background": features_bg, "signal": features_sg}
ax.hist(mjjs.values(), histtype='barstacked', bins=bins_mjj, color=colors, density=True, label=list(mjjs.keys()))
ax.set_xlabel("$M_{JJ}$ (GeV)")
ax.set_ylabel("Amount")
ax.set_xlim(1500,7000)
ax.grid()
color = next(coloriter)
ax.axvline(SR[0], linestyle='--', color=color, label='signal region')
ax.axvline(SR[1], linestyle='--', color=color)
color = next(coloriter)
ax.axvline(SB_LEFT, linestyle='--', color=color, label='side bands')
ax.axvline(SB_RIGHT, linestyle='--', color=color)
ax.legend(loc='upper right')
ax.set_title("Dijet invariant mass")

fig.subplots_adjust(wspace=0.75,hspace=0.3)
ax1 = fig.add_subplot(gs[1,:2])
ax2 = fig.add_subplot(gs[1,2:])
ax1.hist([np.array(file_bg["jet_coords"][:N,0,3]), np.array(file_sn["jet_coords"][:n_s,0,3])], density=True, histtype='barstacked', color=colors, bins=200)
ax2.hist([np.array(file_bg["jet_coords"][:N,1,3]), np.array(file_sn["jet_coords"][:n_s,1,3])], density=True, histtype='barstacked', color=colors, bins=200)
for ax in (ax1,ax2):
    ax.set_xlabel("$m$ [GeV]")
    ax.set_ylabel("rel.freq.")
    ax.set_xlim(0,1000)
ax1.set_title("Jet 1")
ax2.set_title("Jet 2")
#ax.set_yscale('log')
#fig.suptitle("LHCO dataset")
print("Saving")
print(f"SR: 1/1000\nTotal events:\n\tBackground: {features_bg.shape[0]}\n\tSignal: {features_sg.shape[0]/factor}")
in_sr_bg = np.logical_and(features_bg>SR[0], features_bg<SR[1])
in_sr_sg = np.logical_and(features_sg>SR[0], features_sg<SR[1])
print(f"In Signal Region:\n\tBackground: {np.sum(in_sr_bg)}\n\tSignal: {np.sum(in_sr_sg)/factor}")

in_sb_bg = np.logical_and(np.logical_and(features_bg>SB_LEFT, features_bg<SB_RIGHT), ~in_sr_bg)
in_sb_sg = np.logical_and(np.logical_and(features_sg>SB_LEFT, features_sg<SB_RIGHT), ~in_sr_sg)
print("SB cut:")
print(f"In Side Bands:\n\tBackground: {np.sum(in_sb_bg)}\n\tSignal: {np.sum(in_sb_sg)/factor}")

plt.savefig(PLOTS_DIR+"/mjj_pt900-combined.png", bbox_inches='tight', dpi=200)
plt.savefig(PLOTS_DIR+"/mjj_pt900-combined.pgf", bbox_inches='tight')
file_bg.close()
file_sn.close()

x = input("Upload to onedrive? [Y/N]")
if(x.lower() == "y"):
    from onedrive import onedrive
    handler = onedrive.OneDriveHandler()
    response = handler.upload(PLOTS_DIR+"/mjj_pt900-combined.pgf", name="lhco_spectrum_pt900-combined.pgf", to_path="Documenten/thesis/plots")