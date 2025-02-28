#! /usr/env/python3 
import os, sys
sys.path.append(os.getenv("HOME")+"/Analysis/lib")
from tqdm.contrib.concurrent import process_map
import numpy as np
import pandas as pd
import tables as pt
import matplotlib.pyplot as plt
import jlogger as jl
import utils
import h5py 
logger = jl.JLogger(storage_id = "features")
NPROC = 4
DATA_DIR = "/scratch/work/geuskens/data"
PLOTS_DIR = os.getenv("HOME") + "/Analysis/plots"

n = 4+3 #pt,eta,phi,m and tau1..3
PT = lambda x: x*n+0 #x is 0 or 1
ETA = lambda x: x*n+1
PHI = lambda x: x*n+2
TAU1 = lambda x: x*n+4
TAU2 = lambda x: x*n+5
TAU3 = lambda x: x*n+6
M = 2*n

keys = [
    "LHCO_RnD", #reference
    "LHCO_RnD S", #reference
    "DP pt1000 R0.8", #DP run with R_fat=0.8 and no mpi
    "DP pt1000",#DP run with R_fat=1.0 and no mpi
    "MG pt1000" #MG run with R_fat=1.0 and no mpi
]

fig = plt.figure(figsize=(10,14))
(sfig_pt,sfig_eta, sfig_phi, sfig_mjj) = fig.subfigures(4,1,wspace=0.05,hspace=0.5)
axs_pt = sfig_pt.subplots(1,2)
axs_eta = sfig_eta.subplots(1,2)
axs_phi = sfig_phi.subplots(1,2)
ax_mjj = sfig_mjj.subplots(1,1)

phi_bins = np.linspace(0,2*np.pi, 50)
eta_bins = np.linspace(-np.pi,np.pi, 50)
bins_pt = [None,None]
bins_mjj = None

# ----- subjettiness ------
fig2 = plt.figure()
axes = fig2.subplots(2,2)
bins = [[100,100],[100,100]]
for key in keys:
    features = logger.retrieve_logged_data("features", group_name=key)
    color = next(axs_pt[0]._get_lines.prop_cycler)["color"]
    style = dict(histtype='step', density=True, color=color)
    _, bins_pt[0], _ = axs_pt[0].hist(features[:,PT(0)], label=key,
                bins=(bins_pt[0] if bins_pt[0] is not None else utils.calculate_bins(features[:,PT(0)], 100, high=2500)),
                **style)
    _, bins_pt[1], _ = axs_pt[1].hist(features[:,PT(1)],
                bins=(bins_pt[1] if bins_pt[1] is not None else utils.calculate_bins(features[:,PT(1)], 100, high=2500)),
                **style)
    axs_eta[0].hist(features[:,ETA(0)], bins=eta_bins, **style)
    axs_eta[1].hist(features[:,ETA(1)], bins=eta_bins, **style)
    axs_phi[0].hist(features[:,PHI(0)], bins=phi_bins, **style)
    axs_phi[1].hist(features[:,PHI(1)], bins=phi_bins, **style)
    _, bins_mjj, _ = ax_mjj.hist(features[:,M],
                bins=(bins_mjj if bins_mjj is not None else utils.calculate_bins(features[:,M], 100, high=6500)),
                **style)

    # ----- subjettiness ------
    div = lambda x,y: np.divide(x,y, out=np.zeros_like(x), where=y!=0)
    _, bins[0][0], _ = axes[0][0].hist(div(features[:,TAU2(0)],features[:,TAU1(0)]), bins=bins[0][0],
                        **style, label=key)
    _, bins[0][1], _ = axes[0][1].hist(div(features[:,TAU2(1)],features[:,TAU1(1)]), bins=bins[0][1],
                    **style)
    _, bins[1][0], _ = axes[1][0].hist(div(features[:,TAU3(0)],features[:,TAU2(0)]), bins=bins[1][0],
                        **style)
    _, bins[1][1], _ = axes[1][1].hist(div(features[:,TAU3(1)],features[:,TAU2(1)]), bins=bins[1][1],
                        **style)


axs_pt[1].set_ylim(axs_pt[0].get_ylim())
axs_pt[0].set_xlabel("$p^1_T$ (GeV)")
axs_pt[1].set_xlabel("$p^2_T$ (GeV)")
axs_eta[0].set_xlabel("$\eta^1_T$ (GeV)")
axs_eta[1].set_xlabel("$\eta^2_T$ (GeV)")
axs_phi[0].set_xlabel("$\phi^1_T$ (GeV)")
axs_phi[1].set_xlabel("$\phi^2_T$ (GeV)")
ax_mjj.set_xlabel("$m_{12}$ (GeV)")
ax_mjj.set_ylim(0,0.003)
for ax in [x for i in range(2) for x in [axs_pt[i], axs_eta[i], axs_phi[i]]]+[ax_mjj]:
    ax.set_ylabel("Rel. freq.")
    ax.grid()
ratio = (1E5/1E6)
axs_pt[0].text(0.4,0.7, f"signal/background={ratio:.2f}", transform=axs_pt[0].transAxes)
fig.suptitle("Features of two leading jets")
sfig_pt.suptitle("$p_T$")
sfig_eta.suptitle("$\eta$")
sfig_phi.suptitle("$\phi$")
sfig_mjj.suptitle("$m_{jj}$")
sfig_pt.legend(loc='upper right')
#fig.tight_layout()
logger.log_figure(fig, PLOTS_DIR+"/jet_features.png", dpi=250,
            comment="Various plots to compare different datasets")

# ----- subjettiness ------
for i in range(2):
    for j in range(2):
        axes[i][j].set_ylabel("Rel. freq.")
        axes[i][j].grid()
axes[0][0].set_xlabel(r"$\tau_{21},\,j_{1}$")
axes[0][1].set_xlabel(r"$\tau_{21},\,j_{2}$")
axes[1][0].set_xlabel(r"$\tau_{32},\,j_{1}$")
axes[1][1].set_xlabel(r"$\tau_{32},\,j_{2}$")
fig2.suptitle("Nsubjettiness ratios of leading jets")
fig2.legend()
fig2.tight_layout()
logger.log_figure(fig2, PLOTS_DIR+"/jet_features_nsubjet.png", dpi=250,
            comment="Nsubjettiness plots to compare different datasets")