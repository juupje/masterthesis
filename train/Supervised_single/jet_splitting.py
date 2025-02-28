import h5py
import numpy as np
import os
import glob
import shutil

"""
outfile1: Jet1
    4mom: (n_events, n_particles, 4), (E, px, py, pz) for each particle
    coords: (n_events, n_particles, 2), (eta, phi) for each particle
    features: (n_events, n_particles, n_features), features for every particle: (deta, dphi, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR)
    mask: (n_events, n_particles)
    signal: (n_events, 1)
    jet_features: (n_events, 4), (tau1j1, tau2j1, tau3j1, mjj)
    jet_coords: (n_events, 4), (pt, eta, phi, m) for jet 1
outfile2: Jet 2
    4mom: (n_events, n_particles, 4), (E, px, py, pz) for each particle
    coords: (n_events, n_particles, 2), (eta, phi) for each particle
    features: (n_events, n_particles, n_features), features for every particle: (deta, dphi, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR)
    mask: (n_events, n_particles)
    signal: (n_events, 1)
    jet_features: (n_events, 4), (tau1j2, tau2j2, tau3j2, mjj)
    jet_coords: (n_events, 4), (pt, eta, phi, m) for jet 2
"""

IN_DIR = os.path.join(os.getenv("DATA_DIR"), "lhco", "efp", "ordered")
OUT_DIR = os.path.join(os.getenv("DATA_DIR"), "lhco", "efp", "ordered", "singlejet")
OUT_DIR2 = os.path.join(os.getenv("DATA_DIR"), "lhco", "efp", "ordered", "noisejet")
for file in glob.glob(os.path.join(IN_DIR, "*.h5")):
    fname = os.path.basename(file)
    infile = h5py.File(file, "r")
    outfile1 = h5py.File(os.path.join(OUT_DIR, "jet1", fname), "w")
    outfile2 = h5py.File(os.path.join(OUT_DIR, "jet2", fname), "w")
    for key in ["4mom", "coords", "features", "mask"]:
        outfile1.create_dataset(key, data=infile[f"jet1/{key}"])
        outfile2.create_dataset(key, data=infile[f"jet2/{key}"])
    outfile1.create_dataset("signal", data=infile["signal"])
    outfile2.create_dataset("signal", data=infile["signal"])
    #(tau1j1, tau2j1, tau3j1, tau1j2, tau2j2, tau3j2, mjj) for every event
    jet_features = np.array(infile["jet_features"])
    outfile1.create_dataset("jet_features", data=jet_features)
    outfile2.create_dataset("jet_features", data=jet_features)

    jet_coords = np.array(infile["jet_coords"])
    outfile1.create_dataset("jet_coords", data=jet_coords)
    outfile2.create_dataset("jet_coords", data=jet_coords)
    outfile1.close()
    outfile2.close()
    infile.close()

n = []
means = {"jet1": {"N": [], "4mom": [], "coords": [], "features": []}, "jet2":{"N": [], "4mom": [], "coords": [], "features": []}}
#get the means for all the files
for file in glob.glob(os.path.join(IN_DIR, "*.h5")):
    fname = os.path.basename(file)
    shutil.copy2(file, os.path.join(OUT_DIR2, "jet1_"+fname))
    shutil.copy2(file, os.path.join(OUT_DIR2, "jet2_"+fname))
    outfile = h5py.File(os.path.join(OUT_DIR2, "jet1_"+fname), "r+")
    for jet in ["jet1", "jet2"]:
        means[jet]["N"].append(int(np.mean(np.sum(np.array(outfile[f"{jet}/mask"]), axis=(1,2)))))
        for key in ["4mom", "coords","features"]:
            means[jet][key].append(np.mean(np.array(outfile[f"{jet}/{key}"]), axis=0))
    n.append(outfile["signal"].shape[0])
    outfile.close()

#aggregate the means
for jet in ["jet1", "jet2"]:
    for key in ["N", "4mom", "coords","features"]:
        means[jet][key] = np.average(means[jet][key], weights=n, axis=0)

for jet in ["jet1", "jet2"]:
    for file in glob.glob(os.path.join(IN_DIR, f"*.h5")):
        fname = os.path.basename(file)
        print(f"Writing means of {jet} to {fname}")
        outfile = h5py.File(os.path.join(OUT_DIR2, f"{jet}_{fname}"), "r+")
        grp = outfile[jet]
        for key in ["4mom", "coords","features"]:
            data = means[jet][key]
            data[int(means[jet]["N"]):] = 0
            data = np.repeat(np.expand_dims(data,axis=0), grp[key].shape[0], axis=0)
            grp[key][:] = data
        mask = np.zeros_like(grp["mask"])
        mask[:, :int(means[jet]["N"])] = 1
        grp["mask"][:] = mask
        outfile.close()