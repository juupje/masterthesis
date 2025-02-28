import h5py
import multiprocessing as mp
import os, re
import numpy as np
import pandas as pd
from typing import Tuple
from utils.coords import PtEtaPhi_to_EXYZ, XYZ_to_PtEtaPhi

"""
N= #particles
The None index indicates the event axis
File layout:
    - 4mom: (None, N, 4) - 4 momenta stored in (e, px, py, pz) format for every particle.
    - coords: (None, N, 2) - (eta, phi) for every particle
    - features: (None, N, 9) - features for every particle: (delta eta, delta phi, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR)
    - mask: (None, N, 1) - mask (1 if particle is present, 0 otherwise) for each particle
    - jet_coords: (None, 4) - (pt, eta, phi, m) for the jet
    - signal: (None) - signal bit (1 for signal, 0 otherwise)
"""

def prepare_data(events:np.ndarray, pt_jet:np.ndarray, m_jet:np.ndarray, max_n_particles:int):

    events = events.astype(np.float32)
    #DO NOT sort events
    events = events[:, :max_n_particles]

    mask = np.expand_dims(~np.all(events==0,axis=-1),axis=-1).astype(np.int8)

    e, px, py, pz   = PtEtaPhi_to_EXYZ(events[:,:,0], events[:,:,1], events[:,:,2])
    mom4   = np.stack((e, px, py, pz), axis=-1, dtype=np.float32)
    pt, coords = events[:,:,0], events[:,:,1:]
    jet_4mom = np.sum(mom4, axis=1)
    jet_coords = np.zeros((events.shape[0], 4)) #(pt, eta, phi, m)
    jet_coords[:,1:3] = np.stack(XYZ_to_PtEtaPhi(jet_4mom[:,1], jet_4mom[:,2], jet_4mom[:,3])[1:],axis=1)
    jet_coords[:,0] = pt_jet
    jet_coords[:,3] = m_jet
    #DO NOT use the jet 4-momentum to compute the relative quantities
    trunc_jet_coords = np.stack((*XYZ_to_PtEtaPhi(jet_4mom[:,1], jet_4mom[:,2], jet_4mom[:,3]), np.sqrt(np.maximum(0, jet_4mom[:,0]**2-np.sum(np.square(jet_4mom[:,1:]),axis=-1)))),axis=1)

    rel_pt = pt/np.expand_dims(trunc_jet_coords[:,0],axis=1)
    delta_eta = events[:, :, 1]-np.expand_dims(trunc_jet_coords[:,1],axis=1)
    delta_phi = events[:, :, 2]-np.expand_dims(trunc_jet_coords[:,2],axis=1)
    deltaR = np.sqrt(np.square(delta_phi)+np.square(delta_eta))
    
    log_pt = np.log(pt+1e-30)
    log_pt[~mask] = 0

    log_rel_pt = np.log(rel_pt+1e-30)
    log_rel_pt[~mask] = 0

    log_e = np.log(e+1e-30)
    log_e[~mask] = 0

    rel_e = e/np.expand_dims(jet_4mom[:,0],axis=1)
    log_rel_e = np.log(rel_e+1e-30)
    log_rel_e[~mask] = 0

    features = np.stack((delta_eta,delta_phi, log_pt,log_e, rel_pt, rel_e, log_rel_pt, log_rel_e, deltaR),axis=-1)
    return mom4, coords, features, mask, jet_coords

def chunk_processor(infile:h5py.File, in_queue:mp.Queue, out_queue:mp.Queue):
    while (chunk := in_queue.get()) is not None:
        print(f"Processing chunk {chunk}")
        dataset = infile["raw"][chunk[0]:chunk[1]]
        pt_jet = infile["ptj"][chunk[0]:chunk[1]]
        m_jet = infile["m_jet"][chunk[0]:chunk[1]]
        mom4, coords, feats, mask, jet_coords = prepare_data(dataset, pt_jet=pt_jet, m_jet=m_jet, max_n_particles=n_particles)
        out_queue.put((chunk, (mom4, coords, feats, mask, jet_coords)))
    # Signal the writer that this reader is done
    out_queue.put(None)

def setup_file(outfile:h5py.File, n_events, n_particles):
    outfile.create_dataset("4mom", shape=(n_events, n_particles, 4), dtype='float32')
    outfile.create_dataset("coords", shape=(n_events, n_particles, 2), dtype='float32')
    outfile.create_dataset("features", shape=(n_events, n_particles, 9), dtype='float32')
    outfile.create_dataset("mask", shape=(n_events, n_particles, 1), dtype='int8')
    outfile.create_dataset("jet_coords", shape=(n_events, 4), dtype='float32')
    outfile.create_dataset("signal", shape=(n_events,), dtype='int8')

def writer(out_queue:mp.Queue, n_workers:int):
    done_counter = 0
    outfile = h5py.File(args["output"], 'w')
    setup_file(outfile, n_events, n_particles)
    while done_counter < n_workers:
        res = out_queue.get()
        if(res == None):
            done_counter += 1
            print(f"Worker {done_counter} is done!")
            continue
        chunk, (mom4, coords, feats, mask, jet_coords) = res
        print("Writing chunk", chunk)
        outfile["4mom"][chunk[0]:chunk[1]] = mom4
        outfile["coords"][chunk[0]:chunk[1]] = coords
        outfile["features"][chunk[0]:chunk[1]] = feats
        outfile["mask"][chunk[0]:chunk[1]] = mask
        outfile["jet_coords"][chunk[0]:chunk[1]] = jet_coords
        outfile['signal'][chunk[0]:chunk[1]] = args['signal']
    outfile.close()


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("Preprocessing LHCO data")
    parser.add_argument("input", help="Input file", type=str)
    parser.add_argument("output", help="Output file", type=str)
    parser.add_argument("--events", "-e", help="Number of events", type=int)
    parser.add_argument("--offset", "-o", help="Skip this many events at the start of the file", default=0, type=int)
    parser.add_argument("--number", "-N", help="Number of particles", default=-1, type=int)
    parser.add_argument("--n_proc", "-n", help="Number of processes to use", type=int, default=3)
    parser.add_argument("--signal", "-s", help="Signal bit", type=int, choices=[0,1], required=True)
    parser.add_argument("--chunksize", "-c", help="Chunk size", type=int, default=-1)
    args = vars(parser.parse_args())
    
    if not os.path.exists(args["input"]):
        print("Input file does not exist")
        exit(-1)

    infile = h5py.File(args["input"])
    shape = infile["raw"].shape
    
    offset = args["offset"]
    if(args["events"] is None):
        n_events = shape[0]-offset
    else:
        n_events = min(args["events"],shape[0]-offset)
    n_particles_in_file = shape[1]
    n_particles = min(args["number"], n_particles_in_file) if args["number"]>0 else n_particles_in_file

    chunksize = args["chunksize"] if args["chunksize"]>0 else int(n_events/args["n_proc"])
    chunks = [[i*chunksize+offset,min(n_events,(i+1)*chunksize)+offset] for i in range((n_events+chunksize-1)//chunksize)]
    print("Computing chunks:", chunks)
    print("Using n_particles:", n_particles)
    fname_split = os.path.splitext(args["output"])
    assert(fname_split[1]=='.h5')
    filenames = [fname_split[0]+f"-{i:d}.h5" for i in range(len(chunks))]

    if(args["n_proc"]>1):
        in_queue = mp.Queue()
        for chunk in chunks: in_queue.put(chunk)
        for _ in range(args["n_proc"]): in_queue.put(None) #add the terminal signals
        out_queue = mp.Queue()

        processes = []
        for i in range(args["n_proc"]-1):
            process = mp.Process(target=chunk_processor, args=(infile, in_queue, out_queue))
            processes.append(process)
            process.start()

        process = mp.Process(target=writer, args=(out_queue, len(processes)))
        processes.append(process)
        process.start()
        for p in processes:
            p.join()
    else:
        from tqdm import tqdm
        outfile = h5py.File(args["output"], 'w')
        setup_file(outfile, n_events, n_particles)
        for chunk in tqdm(chunks):
            dataset = infile["raw"][chunk[0]:chunk[1]]
            pt_jet = infile["ptj"][chunk[0]:chunk[1]]
            m_jet = infile["m_jet"][chunk[0]:chunk[1]]
            mom4, coords, feats, mask, jet_coords = prepare_data(dataset, pt_jet, m_jet, n_particles)
            outfile["4mom"][chunk[0]:chunk[1]] = mom4
            outfile["coords"][chunk[0]:chunk[1]] = coords
            outfile["features"][chunk[0]:chunk[1]] = feats
            outfile["mask"][chunk[0]:chunk[1]] = mask
            outfile["jet_coords"][chunk[0]:chunk[1]] = jet_coords
        outfile["signal"][:] = args['signal']
        outfile.close()
    infile.close()