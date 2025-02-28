#! /usr/env/python3 
import os, sys
sys.path.append(os.getenv("HOME")+"/Analysis/lib")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jlogger as jl
import utils
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

features = pd.read_hdf(DATA_DIR + "/events_features_v2.h5")
features_bg = features[features["label"]==0].to_numpy(dtype=np.float32)
features_sg = features[features["label"]==1].to_numpy(dtype=np.float32)
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

fig = plt.figure(figsize=(6,5))
ax = fig.subplots(1,1)
bins_mjj = 500
colors = utils.colors.ColorIter()
dfs = {"background": features_bg, "signal": features_sg}
mjjs = {}
for key, df in dfs.items():
    mjjs[key] = m(df[:,PX(0)], df[:,PY(0)], df[:,PZ(0)],df[:,M(0)], df[:,PX(1)], df[:,PY(1)], df[:,PZ(1)], df[:,M(1)])
ax.hist(mjjs.values(), histtype='barstacked', bins=bins_mjj, color=(next(colors),next(colors)), label=list(mjjs.keys()))
ax.set_xlabel("$m_{12}$ (GeV)")
ax.set_ylabel("Amount")
ax.set_xlim(None,7000)
ax.set_ylim(100,None)
ax.grid()
color = next(colors)
ax.axvline(3300, linestyle='--', color=color)
ax.axvline(3700, linestyle='--', color=color)
#ax.set_yscale('log')
fig.suptitle("LHCO dataset")
fig.legend()
print("Saving")
plt.savefig(PLOTS_DIR+"/mjj.png", dpi=250)
