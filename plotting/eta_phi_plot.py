import matplotlib.pyplot as plt
import os, sys
import numpy as np
import utils
import h5py

PLOT_DIR = os.path.join(os.getenv("HOME"), "Analysis", "plots", "pt_900-hists")
DATA_DIR = "/scratch/work/geuskens/data/lhco/new/pt900"
particle_file = "N30-bg-10.h5"

n_events = 50000
particle_coords_j1 = np.empty((n_events, 30, 3))
particle_coords_j2 = np.empty((n_events, 30, 3))

file = h5py.File(os.path.join(DATA_DIR, particle_file))
particle_coords_j1[:,:,1:] = np.array(file["jet1/coords"][:n_events])
particle_coords_j1[:,:,0] = np.sqrt(np.sum(np.square(np.array(file["jet1/4mom"][:n_events])[:,:,1:3]), axis=-1))

particle_coords_j2[:,:,1:] = np.array(file["jet2/coords"][:n_events])
particle_coords_j2[:,:,0] = np.sqrt(np.sum(np.square(np.array(file["jet2/4mom"][:n_events])[:,:,1:3]), axis=-1))
print(particle_coords_j1[0,:,2])
fig, axes = plt.subplots(3,2, figsize=(6,8))
for col, coords in zip((0,1), (particle_coords_j1, particle_coords_j2)):
    for row,name in zip((0,1,2),("p_T", r"\Delta\eta", r"\Delta\phi")):
        axes[row][col].hist(coords[:,:,row].flatten(), bins=1000, histtype='step')
        axes[row][col].hist(coords[:,:,row].flatten(), bins=1000, histtype='step')
        axes[row][col].set_yscale('log')
        axes[row][col].set_title(f"${{{name}}}_{{j{col+1}}}$")
        axes[row][col].set_xlabel(f"${name}$")
        axes[row][col].set_ylabel("freq.")

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "deta_dphi_per_particle.png"), dpi=250)