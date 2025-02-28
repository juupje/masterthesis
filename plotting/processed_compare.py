#! /usr/env/python3
"""
Compares preprocessed datasets (just the jet-features)
"""
import os, sys
sys.path.append(os.getenv("HOME")+"/Analysis/lib")
from tqdm.contrib.concurrent import process_map
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jlogger as jl
import utils
import h5py 
logger = jl.JLogger(storage_id = "features")
NPROC = 4
DATA_DIR = "/scratch/work/geuskens/data/lhco"
PLOTS_DIR = os.getenv("HOME") + "/Analysis/plots"
#(tau1j1, tau2j1, tau3j1, tau1j2, tau2j2, tau3j2, mjj)
n = 3 #tau1, tau2, tau3
TAU1 = lambda x: x*n
TAU2 = lambda x: x*n+1
TAU3 = lambda x: x*n+2
MJJ = 2*n

def pt(x,y):
    return np.sqrt(np.square(x)+np.square(y))
def eta(x,y,z):
    return np.arctanh(z/np.sqrt(np.square(x)+np.square(y)+np.square(z)))
def phi(x,y):
    return np.arctan2(x,y)+np.pi #0-2pi
def m(x1,y1,z1,m1,x2,y2,z2,m2):
    p1_2 = np.square(x1)+np.square(y1)+np.square(z1)
    p2_2 = np.square(x2)+np.square(y2)+np.square(z2)
    E1_2 = np.square(m1)+p1_2
    E2_2 = np.square(m2)+p2_2
    return np.sqrt(E1_2+E2_2+2*np.sqrt(E1_2)*np.sqrt(E2_2)-p1_2-p2_2-2*(x1*x2+y1*y2+z1*z2))

datasets = {"pt500":  "original/Supervised-N30-bg-train.h5",
            "pt900":  "new/Supervised-N30-bg_pt900-total.h5",
            "pt1000": "new/Supervised-N30-bg_pt1000-total.h5"}

fig_mjj = plt.figure()
ax_mjj = fig_mjj.add_subplot(1,1,1)

bins_mjj = 50
m1_bins, m2_bins = 100, 100

# ------- subjettiness --------
fig2 = plt.figure(figsize=(6,9))
axes = fig2.subplots(3,2)
bins = [[50,50],[50,50]]
colors = utils.colors.ColorIter()
for key in datasets:
    color = next(colors)
    style = dict(histtype='step', density=True, color=color)
    file = h5py.File(os.path.join(DATA_DIR,datasets[key]), mode='r')
    features = file["jet_features"]
    m1,m2,mjj = file["jet_coords"][:,0,-1], file["jet_coords"][:,1,-1], features[:,MJJ]
    print(key+":",mjj.shape)
    tau1j1,tau2j1,tau3j1 = features[:,TAU1(0)],features[:,TAU2(0)],features[:,TAU3(0)]
    tau1j2,tau2j2,tau3j2 = features[:,TAU1(1)],features[:,TAU2(1)],features[:,TAU3(1)]
    _, bins_mjj, _ = ax_mjj.hist(mjj, bins=bins_mjj, label=key, **style)

    _, m1_bins, _ = axes[2][0].hist(m1, bins=m1_bins, **style)
    _, m2_bins, _ = axes[2][1].hist(m2, bins=m2_bins, **style)
    # ----- subjettiness ------
    div = lambda x,y: np.divide(x,y, out=np.zeros_like(x), where=y!=0)
    _, bins[0][0], _ = axes[0][0].hist(div(tau2j1,tau1j1), bins=bins[0][0],
                        **style, label=key)
    _, bins[0][1], _ = axes[0][1].hist(div(tau2j2,tau1j2), bins=bins[0][1],
                    **style)
    _, bins[1][0], _ = axes[1][0].hist(div(tau3j1,tau2j1), bins=bins[1][0],
                        **style)
    _, bins[1][1], _ = axes[1][1].hist(div(tau3j2,tau2j2), bins=bins[1][1],
                        **style)
    
ax_mjj.set_xlabel("$m_{12}$ (GeV)")
fig_mjj.suptitle("$m_{jj}$ of two leading jets")
fig_mjj.legend(loc='upper right')
#fig.tight_layout()
logger.log_figure(fig_mjj, PLOTS_DIR+"/preprocessed_mjj_comparison.png", dpi=250,
            comment="MJJ plot to compare different datasets", bbox_inches='tight')

# ----- subjettiness ------
for i in range(len(axes)):
    for j in range(2):
        axes[i][j].set_ylabel("Rel. freq.")
        axes[i][j].grid()
axes[0][0].set_xlabel(r"$\tau_{21},\,j_{1}$")
axes[0][1].set_xlabel(r"$\tau_{21},\,j_{2}$")
axes[1][0].set_xlabel(r"$\tau_{32},\,j_{1}$")
axes[1][1].set_xlabel(r"$\tau_{32},\,j_{2}$")
axes[2][0].set_xlabel(r"$m_{j1}$")
axes[2][1].set_xlabel(r"$m_{j2}$")
fig2.suptitle("Nsubjettiness ratios and masses of leading jets")
fig2.legend()
fig2.tight_layout()
logger.log_figure(fig2, PLOTS_DIR+"/preprocessed_tau+m_comparison.png", dpi=250,
            comment="Nsubjettiness and mass plots to compare different datasets", bbox_inches='tight')