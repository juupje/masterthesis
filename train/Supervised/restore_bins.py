from data_eval_helpers import LoadTrue
import numpy as np
import pandas as pd
import h5py
import os, tqdm
import argparse

def LoadBins(bin_dir:str):
    pt_bins = np.load(os.path.join(bin_dir,'pt_bins_1Mfromeach_403030.npy'))
    eta_bins = np.load(os.path.join(bin_dir,'eta_bins_1Mfromeach_403030.npy'))
    phi_bins = np.load(os.path.join(bin_dir,'phi_bins_1Mfromeach_403030.npy'))
    return pt_bins,eta_bins,phi_bins

def SaveSamples(continues_jets,mj,ptj,output_path):
    hf = h5py.File(output_path, 'w')
    hf.create_dataset('raw', data=continues_jets)
    hf.create_dataset('m_jet', data=mj)
    hf.create_dataset('ptj', data=ptj)
    hf.close()
    return

def CreateDataFile(output_path, n_events, n_particles):
    hf = h5py.File(output_path, 'w')
    hf.create_dataset("raw", shape=(n_events, n_particles, 3), dtype='float32')
    hf.create_dataset("m_jet", shape=(n_events,), dtype='float32')
    hf.create_dataset("ptj", shape=(n_events,), dtype='float32')
    return hf

def make_continuous(jets, mask,pt_bins,eta_bins,phi_bins, noise=False):
    pt_disc = jets[:, :, 0]
    eta_disc = jets[:, :, 1]
    phi_disc = jets[:, :, 2]

    if noise:
        pt_con = (pt_disc - np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * (
            pt_bins[1] - pt_bins[0]
        ) + pt_bins[0]
        eta_con = (eta_disc - np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * (
            eta_bins[1] - eta_bins[0]
        ) + eta_bins[0]
        phi_con = (phi_disc - np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * (
            phi_bins[1] - phi_bins[0]
        ) + phi_bins[0]
    else:
        pt_con = (pt_disc - 0.5) * (pt_bins[1] - pt_bins[0]) + pt_bins[0]
        eta_con = (eta_disc - 0.5) * (eta_bins[1] - eta_bins[0]) + eta_bins[0]
        phi_con = (phi_disc - 0.5) * (phi_bins[1] - phi_bins[0]) + phi_bins[0]


    pt_con = np.exp(pt_con)
    pt_con[mask] = 0.0
    eta_con[mask] = 0.0
    phi_con[mask] = 0.0
    
    pxs = np.cos(phi_con) * pt_con
    pys = np.sin(phi_con) * pt_con
    pzs = np.sinh(eta_con) * pt_con
    es = (pxs ** 2 + pys ** 2 + pzs ** 2) ** (1. / 2)

    pxj = np.sum(pxs, -1)
    pyj = np.sum(pys, -1)
    pzj = np.sum(pzs, -1)
    ej = np.sum(es, -1)
    
    ptj = np.sqrt(pxj**2 + pyj**2)
    mj = (ej ** 2 - pxj ** 2 - pyj ** 2 - pzj ** 2) ** (1. / 2)

    continuous_jets = np.stack((pt_con, eta_con, phi_con), -1)

    return continuous_jets, ptj, mj

def LoadTrue(discrete_truedata_filename, pt_bins, eta_bins, phi_bins,chunk:slice=None):
    if chunk is not None:
        tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", start=chunk.start, stop=chunk.stop)
    else:
        tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", stop=None)
    tmp = tmp.to_numpy().reshape(len(tmp), -1, 3)
    tmp=tmp[:,:,:]
    mask = tmp[:, :, 0] == -1
    jets_true,ptj_true,mj_true = make_continuous(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)
    return jets_true,ptj_true,mj_true


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file", type=str)
    parser.add_argument("output", help="Output file", type=str)
    parser.add_argument("--n_samples", help="Number of samples to process", type=int, default=-1)
    parser.add_argument("--chunksize", help="Chunksize", type=int, default=8192)
    parser.add_argument("--bin-dir", help="Directory of the bin files", type=str,  required=True)
    args = vars(parser.parse_args())

    f = h5py.File(args["input"], 'r')
    particle_dim = f['discretized/block0_values'].shape[1]
    n_samples = args.get("n_samples", -1)
    if n_samples < 0:
        n_samples = f['discretized/block0_values'].shape[0]
    f.close()

    print("Loading bins")
    pt_bins,eta_bins,phi_bins=LoadBins(args["bin_dir"])
    print("Setting up output file")
    ofile = CreateDataFile(args["output"], n_samples, particle_dim//3)

    for chunk in tqdm.tqdm(range(0, n_samples, args["chunksize"])):
        sl = slice(chunk, min(chunk+args["chunksize"], n_samples))
        jets,ptj,mj=LoadTrue(args["input"],pt_bins,eta_bins,phi_bins,chunk=sl)
        ofile['raw'][sl] = jets
        ofile['m_jet'][sl] = mj
        ofile['ptj'][sl] = ptj
    ofile.close()
