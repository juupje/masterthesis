import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils.colors import ColorIter 
import steganologger

DATA_DIR = os.path.join(os.getenv("DATA_DIR"), "lhco")
files = {"Processed": os.path.join(DATA_DIR, "original", "N30.h5"), "Processed - new": os.path.join(DATA_DIR, "new", "DP_N30.h5")}

"Reference file"
features = os.path.join(DATA_DIR,"raw", "features.h5")
bg_sn_cut_idx = 1_000_000
n_bg = 100_000
n_sn = 100_000

fig = plt.figure(figsize=(12,12))
col_iter = ColorIter()

colors = {key:{"bg":next(col_iter), "sn":next(col_iter)} for key in files}
colors["Reference"] = {"bg":next(col_iter), "sn":next(col_iter)}
hist_style = dict(histtype='step', density=True)

v_idx = 0
h_idx = 1
nrows = 3
ncols = 4
ax_pt1 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_pt1.set_title("$p_T$ jet 1")
ax_pt1.set_xlabel("$p^1_T$ [GeV]")
ax_pt1.set_xlim(0,2500)
ax_pt1.set_ylim(0,0.0055)
h_idx += 1
ax_pt2 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_pt2.set_title("$p_T$ jet 2")
ax_pt2.set_xlabel("$p^2_T$ [GeV]")
ax_pt2.set_xlim(0,2500)
ax_pt2.set_ylim(0,0.0055)
h_idx += 1
ax_m1 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_m1.set_title("$m_j$ jet 1")
ax_m1.set_xlabel("$m^1$ [GeV]")
ax_m1.set_xlim(0,1200)
ax_m1.set_ylim(0,0.0075)
h_idx += 1
ax_m2 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_m2.set_title("$m_j$ jet 2")
ax_m2.set_xlabel("$m^2$ [GeV]")
ax_m2.set_xlim(0,1200)
ax_m2.set_ylim(0,0.0075)

v_idx += 1
h_idx = 1
ax_nsub21_j1 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_nsub21_j1.set_title(r"$\tau_2/\tau_1$ jet 1")
ax_nsub21_j1.set_xlabel(r"$\tau^1_2/\tau^1_1$")
h_idx += 1
ax_nsub21_j2 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_nsub21_j2.set_title(r"$\tau_2/\tau_1$ jet 2")
ax_nsub21_j2.set_xlabel(r"$\tau^2_2/\tau^2_1$")
h_idx += 1
ax_nsub32_j1 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_nsub32_j1.set_title(r"$\tau_3/\tau_2$ jet 1")
ax_nsub32_j1.set_xlabel(r"$\tau^1_3/\tau^1_2$")
#ax_nsub32_j1.set_xlim(0,2)
h_idx += 1
ax_nsub32_j2 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_nsub32_j2.set_title(r"$\tau_3/\tau_2$ jet 2")
ax_nsub32_j2.set_xlabel(r"$\tau^2_3/\tau^2_2$")
#ax_nsub32_j2.set_xlim(0,2)

v_idx += 1
h_idx = 1
ax_np1 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_np1.set_title("#particles jet 1")
ax_np1.set_xlabel("number")
h_idx += 1
ax_np2 = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_np2.set_title("#particles jet 2")
ax_np2.set_xlabel("number")
h_idx += 1
ax_mjj = fig.add_subplot(nrows,ncols, ncols*v_idx+h_idx)
ax_mjj.set_title("$m_{jj}$")
ax_mjj.set_xlabel("$m_{jj}$ [GeV]")
ax_mjj.set_ylim(0,0.003)

div = lambda x,y: np.divide(x,y, out=np.zeros_like(x), where=y!=0)
between = lambda x, a, b: x[(x>=a) & (x<=b)]
bins = {}
for key in files:
    file = h5py.File(files[key], 'r')
    signal = np.array(file["signal"])==1
    background = ~signal
    for name,idx in zip(("bg", "sn"), (background, signal)):
        color = colors[key][name]
        #pt's
        _, bins["pt1"], _ = ax_pt1.hist(file["jet_coords"][idx,0,0], bins=bins.get("pt1", 100), color=color, **hist_style)
        _, bins["pt2"], _ = ax_pt2.hist(file["jet_coords"][idx,1,0], bins=bins.get("pt2", 100), color=color, **hist_style)
        #masses
        _, bins["m1"], _ = ax_m1.hist(file["jet_coords"][idx,0,3],  bins=bins.get("m1", 100), color=color, **hist_style)
        _, bins["m2"], _ =ax_m2.hist(file["jet_coords"][idx,1,3],  bins=bins.get("m1", 100), color=color, **hist_style)
        #tau2/tau1
        _, bins["nsub21_j1"], _ = ax_nsub21_j1.hist(div(file["jet_features"][idx,1],file["jet_features"][idx,0]), bins=bins.get("nsub21_j1", 100), color=color, **hist_style)
        _, bins["nsub21_j2"], _ = ax_nsub21_j2.hist(div(file["jet_features"][idx,1+3],file["jet_features"][idx,0+3]), bins=bins.get("nsub21_j2", 100), color=color, **hist_style)
        #tau3/tau2
        _, bins["nsub32_j1"], _ = ax_nsub32_j1.hist(div(file["jet_features"][idx,2],file["jet_features"][idx,1]), bins=bins.get("nsub32_j1", 100), color=color, **hist_style)
        _, bins["nsub32_j2"], _ = ax_nsub32_j2.hist(div(file["jet_features"][idx,2+3],file["jet_features"][idx,1+3]), bins=bins.get("nsub32_j2", 100), color=color, **hist_style)
        # number of particles
        print(file["jet1/mask"][idx].shape, np.sum(file["jet1/mask"][idx],axis=(1,2)).shape)
        _, bins["np1"], _ = ax_np1.hist(np.sum(file["jet1/mask"][idx],axis=(1,2)), bins=bins.get("np1", 100), color=color, **hist_style)
        _, bins["np2"], _ = ax_np2.hist(np.sum(file["jet2/mask"][idx],axis=(1,2)), bins=bins.get("np2", 100),color=color, **hist_style)
        # mjj
        _, bins["mjj"], _ = ax_mjj.hist(file["jet_features"][idx,-1], bins=bins.get("mjj", 100), color=color, label=f"{key} - {name}", **hist_style)        

features = pd.read_hdf(features)
n = 4+3 #px,py,pz,m and tau1..3
PX = lambda x: x*n+0 #x is 0 or 1
PY = lambda x: x*n+1
PZ = lambda x: x*n+2
M = lambda x: x*n+3
TAU1 = lambda x: x*n+4
TAU2 = lambda x: x*n+5
TAU3 = lambda x: x*n+6
def m(x1,y1,z1,m1,x2,y2,z2,m2):
    p1_2 = np.square(x1)+np.square(y1)+np.square(z1)
    p2_2 = np.square(x2)+np.square(y2)+np.square(z2)
    E1_2 = np.square(m1)+p1_2
    E2_2 = np.square(m2)+p2_2
    return np.sqrt(E1_2+E2_2+2*np.sqrt(E1_2)*np.sqrt(E2_2)-p1_2-p2_2-2*(x1*x2+y1*y2+z1*z2))

pt = lambda x, y: np.sqrt(np.square(x)+np.square(y))
features_bg = features[features["label"]==0].iloc[:n_bg].to_numpy(dtype=np.float32)
features_sg = features[features["label"]==1].iloc[:n_sn].to_numpy(dtype=np.float32)
print(features_bg.shape)
for name, data in zip(("bg", "sn"), (features_bg, features_sg)):
    color = colors["Reference"][name]
    #plot the reference features
    print(data[:,PX(0)].shape)
    _, bins["pt1"], _ = ax_pt1.hist(pt(data[:,PX(0)], data[:,PY(0)]), bins=bins.get("pt1", 100), color=color, **hist_style)
    _, bins["pt2"], _ = ax_pt2.hist(pt(data[:,PX(1)], data[:,PY(1)]), bins=bins.get("pt2", 100), color=color, **hist_style)
    #masses
    _, bins["m1"], _ = ax_m1.hist(data[:,M(0)],  bins=bins.get("m1", 100), color=color, **hist_style)
    _, bins["m2"], _ =ax_m2.hist(data[:,M(1)],  bins=bins.get("m1", 100), color=color, **hist_style)
    #tau2/tau1
    _, bins["nsub21_j1"], _ = ax_nsub21_j1.hist(div(data[:,TAU2(0)],data[:,TAU1(0)]), bins=bins.get("nsub21_j1", 100), color=color, **hist_style)
    _, bins["nsub21_j2"], _ = ax_nsub21_j2.hist(div(data[:,TAU2(1)],data[:,TAU1(1)]), bins=bins.get("nsub21_j2", 100), color=color, **hist_style)
    #tau3/tau2
    _, bins["nsub32_j1"], _ = ax_nsub32_j1.hist(div(data[:,TAU3(0)],data[:,TAU2(0)]), bins=bins.get("nsub32_j1", 100), color=color, **hist_style)
    _, bins["nsub32_j2"], _ = ax_nsub32_j2.hist(div(data[:,TAU3(1)],data[:,TAU2(1)]), bins=bins.get("nsub32_j2", 100), color=color, **hist_style)
    # mjj
    _, bins["mjj"], _ = ax_mjj.hist(m(data[:,PX(0)], data[:,PY(0)], data[:,PZ(0)], data[:,M(0)], data[:,PX(1)], data[:,PY(1)], data[:,PZ(1)], data[:,M(1)]), bins=100, color=color, label=f"Reference - {name}", **hist_style)        

fig.tight_layout()
fig.legend()
fig.savefig("visualization.pdf", dpi=250)
steganologger.encode("visualization.pdf", data=files, overwrite=True)