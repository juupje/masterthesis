from matplotlib import pyplot as plt
import numpy as np
import jlogger as jl
import os
np.random.seed(1)

logger = jl.JLogger("topk_pt")

PLOT_DIR = os.getenv("HOME") + "/Analysis/plots"
N=10000

events = logger.retrieve_logged_data("particles")
jet_idx = logger.retrieve_logged_data("jet_indices")

signal_bit = events[:,300]

def is_sorted(pt):
    for i in range(1,len(pt)):
        if(pt[i]>pt[i-1]):
            return False
    return True

def process(signal):
    selection = np.sort(np.random.choice(np.where(signal_bit==signal)[0], N, replace=False))
    assert(np.alltrue(events[selection,300]==signal))
    pts = events[selection,:-1:3]
    jets = jet_idx[selection]
    jet1_idx = jets==1
    jet2_idx = jets==2
    
    jet1_pts = np.zeros_like(pts, dtype=np.float32)
    jet2_pts = np.zeros_like(pts, dtype=np.float32)

    jet1_pts[jet1_idx] = pts[jet1_idx]
    jet1_pts = np.sort(jet1_pts, axis=1)[:,::-1]
    jet2_pts[jet2_idx] = pts[jet2_idx]
    jet2_pts = np.sort(jet2_pts, axis=1)[:,::-1]

    cumpt1 = np.cumsum(jet1_pts,axis=1)
    print(cumpt1.shape)
    cumpt1 = (cumpt1.T/cumpt1[:,-1]).T #normalize each row with broadcasting magic
    mean1, std1 = np.mean(cumpt1,axis=0), np.std(cumpt1, axis=0)

    cumpt2 = np.cumsum(jet2_pts,axis=1)
    print(cumpt2.shape)
    cumpt2 = (cumpt2.T/cumpt2[:,-1]).T #normalize each row with broadcasting magic
    mean2, std2 = np.mean(cumpt2,axis=0), np.std(cumpt2, axis=0)
    return mean1, std1, mean2, std2, cumpt1[0], cumpt2[0]

fig,(axb, axs) = plt.subplots(1,2,sharex=True,sharey=True)
mean1, std1, mean2, std2, exb1, exb2 = process(signal=0)
x = np.arange(len(mean1))
axb.plot(x, mean1, color='C0', label="Jet1")
axb.fill_between(x, mean1-std1, mean1+std1, color='C0', alpha=0.3)
axb.plot(x, mean2, color='C1', label="Jet2")
axb.fill_between(x, mean2-std2, mean2+std2, color='C1', alpha=0.3)
axb.set_title("Background")

mean1, std1, mean2, std2, exs1, exs2 = process(signal=1)
axs.plot(x, mean1, color='C0', label="Jet1")
axs.fill_between(x, mean1-std1, mean1+std1, color='C0', alpha=0.3)
axs.plot(x, mean2, color='C1', label="Jet2")
axs.fill_between(x, mean2-std2, mean2+std2, color='C1', alpha=0.3)
axs.set_title("Signal")

for ax in (axb,axs):
    ax.legend()
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Rel.Cum. $p_T$ of particles")
    ax.grid()
    ax.set_xlim(0,100)
    ax.set_ylim(0,1)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR,"topk_cumpt.png"), dpi=250)

fig,(axb, axs) = plt.subplots(1,2,sharex=True,sharey=True)
axb.plot(x, exb1, color='C0', label="Jet1")
axb.plot(x, exb2, color='C1', label="Jet2")
axb.set_title("Background")

axs.plot(x, exs1, color='C0', label="Jet1")
axs.plot(x, exs2, color='C1', label="Jet2")
axs.set_title("Signal")

for ax in (axb,axs):
    ax.legend()
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Rel.Cum. $p_T$ of particles")
    ax.grid()
    ax.set_xlim(0,100)
    ax.set_ylim(0,1)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR,"topk_cumpt_single.png"), dpi=250)