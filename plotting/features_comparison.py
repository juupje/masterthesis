#! /usr/env/python3
"""
Compares LHCO feature datasets
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
DATA_DIR = "/scratch/work/geuskens/data"
PLOTS_DIR = os.getenv("HOME") + "/Analysis/plots"

n = 4+3 #px,py,pz,m and tau1..3
PX = lambda x: x*n+0 #x is 0 or 1
PY = lambda x: x*n+1
PZ = lambda x: x*n+2
M = lambda x: x*n+3
TAU1 = lambda x: x*n+4
TAU2 = lambda x: x*n+5
TAU3 = lambda x: x*n+6
MJJ = 2*n

bg = logger.retrieve_logged_data("features", group_name="LHCO_RnD")
sg = logger.retrieve_logged_data("features", group_name="LHCO_RnD S")

features = pd.read_hdf(DATA_DIR + "/events_features_v2.h5")
features_bg = features[features["label"]==0].iloc[:len(bg)].to_numpy(dtype=np.float32)
features_sg = features[features["label"]==1].iloc[:len(sg)].to_numpy(dtype=np.float32)
del features

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

fig = plt.figure(figsize=(10,14))
(sfig_pt,sfig_eta, sfig_phi, sfig_mjj) = fig.subfigures(4,1,wspace=0.05,hspace=0.5)
axs_pt = sfig_pt.subplots(1,2)
axs_eta = sfig_eta.subplots(1,2)
axs_phi = sfig_phi.subplots(1,2)
ax_mjj = sfig_mjj.subplots(1,2)[0]

phi_bins = np.linspace(0,2*np.pi, 50)
eta_bins = np.linspace(-np.pi,np.pi, 50)
bins_pt = [100,100]
bins_mjj = 100

# ----- subjettiness ------
fig2 = plt.figure()
axes = fig2.subplots(2,2)
bins = [[100,100],[100,100]]
for df,lbl in zip([features_bg, features_sg],["Features BG", "Features SG"]):
    color = next(axs_pt[0]._get_lines.prop_cycler)["color"]
    style = dict(histtype='step', density=True, color=color)
    pt1, eta1, phi1 = utils.XYZ_to_PtEtaPhi(df[:,PX(0)], df[:,PY(0)], df[:,PZ(0)])
    pt2, eta2, phi2 = utils.XYZ_to_PtEtaPhi(df[:,PX(1)], df[:,PY(1)], df[:,PZ(1)])
    m1,m2,mjj = df[:,M(0)], df[:,M(1)], df[:,MJJ]
    tau1j1,tau2j1,tau3j1 = df[:,TAU1(0)],df[:,TAU2(0)],df[:,TAU3(0)]
    tau1j2,tau2j2,tau3j2 = df[:,TAU1(1)],df[:,TAU2(1)],df[:,TAU3(1)]
    if("Features" in lbl):
        mjj = m(df[:,PX(0)], df[:,PY(0)], df[:,PZ(0)],df[:,M(0)], df[:,PX(1)], df[:,PY(1)], df[:,PZ(1)], df[:,M(1)])
    _, bins_pt[0], _ = axs_pt[0].hist(pt1, bins=bins_pt[0], **style, label=lbl)
    _, bins_pt[1], _ = axs_pt[1].hist(pt2, bins=bins_pt[1], **style)
    axs_eta[0].hist(eta1, bins=eta_bins, **style)
    axs_eta[1].hist(eta2, bins=eta_bins, **style)
    axs_phi[0].hist(phi1, bins=phi_bins, **style)
    axs_phi[1].hist(phi2, bins=phi_bins, **style)
    _, bins_mjj, _ = ax_mjj.hist(mjj, bins=bins_mjj, **style)

    # ----- subjettiness ------
    div = lambda x,y: np.divide(x,y, out=np.zeros_like(x), where=y!=0)
    _, bins[0][0], _ = axes[0][0].hist(div(tau2j1,tau1j1), bins=bins[0][0],
                        **style, label=lbl)
    _, bins[0][1], _ = axes[0][1].hist(div(tau2j2,tau1j2), bins=bins[0][1],
                    **style)
    _, bins[1][0], _ = axes[1][0].hist(div(tau3j1,tau2j1), bins=bins[1][0],
                        **style)
    _, bins[1][1], _ = axes[1][1].hist(div(tau3j2,tau2j2), bins=bins[1][1],
                        **style)


axs_pt[1].set_ylim(axs_pt[0].get_ylim())
axs_pt[0].set_xlabel("$p^1_T$ (GeV)")
axs_pt[0].set_xlim(None,2500)
axs_pt[1].set_xlim(None,2500)
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
logger.log_figure(fig, PLOTS_DIR+"/features_comparison.png", dpi=250,
            comment="Various plots to compare different datasets", bbox_inches='tight')

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
logger.log_figure(fig2, PLOTS_DIR+"/features_comparison_tau.png", dpi=250,
            comment="Nsubjettiness plots to compare different datasets", bbox_inches='tight')