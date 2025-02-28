from matplotlib import pyplot as plt
import numpy as np
import jlogger as jl
import h5py
import os
np.random.seed(0)

PLOT_DIR = os.getenv("HOME") + "/Analysis/plots"

logger = jl.JLogger("topk_pt")
events = logger.retrieve_logged_data("particles")
jet_idx = logger.retrieve_logged_data("jet_indices")

signal_bit = events[:,300]
bg_events = events[signal_bit==0]
sn_events = events[signal_bit==1]
#get 10 random events
N_events = 8
idxs = np.random.choice(bg_events.shape[0], N_events, replace=False)
idxs_s = np.random.choice(sn_events.shape[0], N_events, replace=False)
fig,(ax1,ax2) = plt.subplots(1,2, sharey=True)
offset = np.linspace(0,0.6,N_events)-0.6/2
for i,idx in enumerate(idxs):
    ax1.scatter(jet_idx[idx]+offset[i], bg_events[idx,:-1:3],s=4)
for i,idx in enumerate(idxs_s):
    ax2.scatter(jet_idx[idx]+offset[i], sn_events[idx,:-1:3],s=4)
    
for ax in (ax1, ax2):
    ax.set_ylabel("$p_T$ [GeV]")
    ax.set_yscale("log")
    #ax.grid()
    ax.axvline(0.5, color='k', linestyle='-', alpha=0.5)
    ax.axvline(1.5, color='k', linestyle='-', alpha=0.5)
    ax.set_xticks([0,1,2], labels=["Other", "Jet 1", "Jet 2"])
ax1.set_title("Background")
ax2.set_title("Signal")
fig.tight_layout()
logger.log_figure(fig, os.path.join(PLOT_DIR, "topk_pt_per_jet.png"), dpi=250,
                    comment="See to which jets the top 100 particles according to pT belong")