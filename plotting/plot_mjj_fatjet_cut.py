#! /usr/env/python3
#Copy of plot_mjj.py, but for self-generated data
import os, sys
sys.path.append(os.getenv("HOME")+"/Analysis/lib")
import numpy as np
import h5py
import matplotlib.pyplot as plt
import jlogger as jl
import utils
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from utils.calc import linreg

def exp_dist(x,b,l):
    return np.exp(l*x)*b


DATA_DIR = "/scratch/work/geuskens/data/lhco/new/"
PLOTS_DIR = os.getenv("HOME") + "/Analysis/plots/data_gen"

SR = (3300,3700)
SB_LEFT = 2900
SB_RIGHT = 4100

do_combine = False

pt_cut = 850
file_bg = h5py.File(os.path.join(DATA_DIR, "pt900/N100-bg.h5"),'r')
file_bg500 = h5py.File(os.path.join(DATA_DIR, "pt500/N100-bg_fatjet-cut.h5"),'r')
file_sn = h5py.File(os.path.join(DATA_DIR, "pt900/N100-sn.h5"),'r')
features_bg = np.array(file_bg["jet_features"][:,-1])
n_s = int(1e-3*features_bg.shape[0])
features_sg = np.array(file_sn["jet_features"][:n_s,-1])
pt1 = np.array(file_bg["jet_coords"][:,0,0])
pt2 = np.array(file_bg["jet_coords"][:,1,0])

pt_500 = np.array(file_bg500["jet_coords"][:,:,0])
max_pt = np.max(pt_500, axis=1)
selected = max_pt>pt_cut
print(f"selected {np.sum(selected)} of {selected.shape[0]} events")
selected = np.where(selected)[0]
pt1_500 = pt_500[selected,0]
pt2_500 = pt_500[selected,1]
idx = np.zeros_like(max_pt, dtype=bool)
features_bg500 = np.array(file_bg500["jet_features"][:,-1])[selected]
print("test4")

print("Plotting")
gs = gridspec.GridSpec(2,4)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(gs[0,:])
bins_mjj = np.linspace(1000,7000, 601)
colors = utils.colors.ColorIter()
mjjs = {"background": features_bg, "signal": features_sg}
count1, _, _= ax.hist(mjjs.values(), histtype='barstacked', bins=bins_mjj, color=(next(colors), next(colors)), label=list(mjjs.keys()))
count2, _, _= ax.hist(features_bg500, histtype='step', bins=bins_mjj, color=next(colors), label="background new")

print("Fitting")
#curve fitting
#original data
idx = (bins_mjj>2900) & (bins_mjj < 4100)
x = bins_mjj[idx]
x = (x[:-1]+x[1:])/2
y = count1[0][idx[:-1]][:-1]
popt_origional, pcov = curve_fit(exp_dist, x, y, sigma=np.sqrt(y), absolute_sigma=True, p0=(1,-1e-3), bounds=((0,-1), (np.inf, 0)))
print(popt_origional, pcov)
ax.plot(bins_mjj, exp_dist(bins_mjj, *popt_origional), label='fit')

#sum
count_sum, _, _ = ax.hist(bins_mjj[:-1]+(bins_mjj[1]-bins_mjj[0])/2, bins=bins_mjj, weights=count1[0]+count2, histtype='step', color=next(colors), label="sum")

def sum_dist(x):
    #find the correct bin
    bin_idx = np.floor((x-bins_mjj[0])/(bins_mjj[1]-bins_mjj[0])).astype(np.int16)
    max_bin = count_sum.shape[0]
    counts = np.concatenate((count_sum, [0]), axis=0)
    bin_idx[(bin_idx<0) | (bin_idx>=max_bin)] = max_bin
    return counts[bin_idx]

def dist_500(x):
    #find the correct bin
    bin_idx = np.floor((x-bins_mjj[0])/(bins_mjj[1]-bins_mjj[0])).astype(np.int16)
    max_bin = count2.shape[0]
    counts = np.concatenate((count2, [0]), axis=0)
    bin_idx[(bin_idx<0) | (bin_idx>=max_bin)] = max_bin
    return counts[bin_idx]

max_freq = exp_dist(2900, *popt_origional)
def target(x, cutoff=2000):
    t = np.where(x>2900, exp_dist(x,*popt_origional), max_freq)
    t[x<cutoff] = 0
    return t
#first do the original data
r = np.random.random(features_bg.shape[0])*sum_dist(features_bg)
t = target(features_bg, cutoff=2700)
idx_selected1 = (r < t)
print(f"Selected {np.sum(idx_selected1):d} events from the original data")

r = np.random.random(features_bg500.shape[0])
above_threshold = features_bg500>2700
r[above_threshold] *= sum_dist(features_bg500[above_threshold])
r[~above_threshold] *= dist_500(features_bg500[~above_threshold])
t = target(features_bg500, cutoff=2000)
idx_selected2 = r < t
print(f"Selected {np.sum(idx_selected2):d} events from the new data")
data = {"mjj":np.concatenate((features_bg[idx_selected1],features_bg500[idx_selected2]),axis=0),
        "pt1":np.concatenate((pt1[idx_selected1],pt1_500[idx_selected2]),axis=0),
        "pt2":np.concatenate((pt2[idx_selected1],pt2_500[idx_selected2]),axis=0)}
ax.hist(data["mjj"], bins=bins_mjj, color=next(colors), alpha=0.7,label="sampled")

ax.set_xlabel("$m_{12}$ (GeV)")
ax.set_ylabel("Amount")
ax.set_xlim(None,7000)
ax.set_ylim(100,35000)
ax.grid() 
color = next(colors)
ax.axvline(SR[0], linestyle='--', color=color, label='signal region')
ax.axvline(SR[1], linestyle='--', color=color)
color = next(colors)
ax.axvline(SB_LEFT, linestyle='--', color=color, label='side bands')
ax.axvline(SB_RIGHT, linestyle='--', color=color)
ax.legend(loc='upper right')#, bbox_to_anchor=(1,1))


ax1 = fig.add_subplot(gs[1,:2])
ax2 = fig.add_subplot(gs[1,2:])
c1 = next(colors)
c2 = next(colors)
c3 = next(colors)
ax1.hist(pt1, histtype='step', bins=400, color=c1, density=True, label="jet1")
ax1.hist(pt1_500, histtype='step', bins=400, color=c2, density=True, label="jet1 new")
ax1.hist(data["pt1"], histtype='step', bins=400, color=c3, density=True,label="jet1 combined")
ax2.hist(pt2, histtype='step', bins=400, color=c1, density=True,label="jet2")
ax2.hist(pt2_500, histtype='step', bins=400, color=c2, density=True,label="jet2 new")
ax2.hist(data["pt2"], histtype='step', bins=400, color=c3, density=True,label="jet2 combined")
ax1.set_xlim(None,3000)
ax2.set_xlim(None,3000)
#ax.set_yscale('log')
ax1.legend()
fig.suptitle("LHCO dataset")
fig.subplots_adjust(wspace=0.75)
print("Saving")
print(f"SR: 1/1000\nTotal events:\n\tBackground: {features_bg.shape[0]}\n\tSignal: {features_sg.shape[0]}")
in_sr_bg = np.logical_and(features_bg>SR[0], features_bg<SR[1])
in_sr_sg = np.logical_and(features_sg>SR[0], features_sg<SR[1])
print(f"In Signal Region:\n\tBackground: {np.sum(in_sr_bg)}\n\tSignal: {np.sum(in_sr_sg)}")

in_sb_bg = np.logical_and(np.logical_and(features_bg>SB_LEFT, features_bg<SB_RIGHT), ~in_sr_bg)
in_sb_sg = np.logical_and(np.logical_and(features_sg>SB_LEFT, features_sg<SB_RIGHT), ~in_sr_sg)
print("SB cut:")
print(f"In Side Bands:\n\tBackground: {np.sum(in_sb_bg)}\n\tSignal: {np.sum(in_sb_sg)}")

plt.savefig(PLOTS_DIR+f"/pt500/mjj+pt_cut={pt_cut}.png", dpi=250)

if(do_combine):
    from utils.calc import create_chunks
    chunksize = 50_000
    def copy_dataset(ofile:h5py.File, name:str, obj1:h5py.Dataset):
        if(type(obj1) is not h5py.Dataset): return
        print(name)
        obj2 = file_bg500[name]
        chunks1 = create_chunks(chunksize, start=0, total_size=obj1.shape[0])
        chunks2 = create_chunks(int(chunksize*obj2.shape[0]/obj1.shape[0]), start=0, total_size=obj2.shape[0])
        print(len(chunks1), len(chunks2))
        count = 0
        for chunk1,chunk2 in zip(chunks1, chunks2):
            print(chunk1,"+",chunk2)
            data1 = np.array(obj1[chunk1[0]:chunk1[1],...])
            data1 = data1[idx_selected1[chunk1[0]:chunk1[1]]]
            data2 = np.array(obj2[chunk2[0]:chunk2[1],...])
            data2 = data2[idx_selected2[chunk2[0]:chunk2[1]]]
            data = np.concatenate((data1,data2), axis=0)
            np.random.shuffle(data)
            ofile[name][count:count+data.shape[0],...] = data
            count += data.shape[0]
        print(f"Copied {name}: {obj1.shape}+{obj2.shape} -> ({0:d}:{count:d}, {', '.join((str(x) for x in obj1.shape[1:]))})")
    
    output_file = h5py.File(os.path.join(DATA_DIR, "N100-bg-combined.h5"), 'w')
    total1 = np.sum(idx_selected1)
    total2 = np.sum(idx_selected2)
    total = total1+total2

    def create(name, obj):
        if type(obj) is h5py.Dataset:
            output_file.create_dataset(name, shape=(total,*obj.shape[1:]))
    file_bg.visititems(lambda name, obj: create(name, obj))
    file_bg.visititems(lambda name, obj: copy_dataset(output_file, name, obj))
 
file_bg500.close()
file_bg.close()
file_sn.close()