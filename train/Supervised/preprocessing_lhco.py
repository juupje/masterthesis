import h5py
import tqdm
from utils.calc import get_nparticles, create_chunks
from utils.coords import PtEtaPhi_to_EXYZ
import os, re
import numpy as np
import pandas as pd
import fastjet as fj
import pyjettiness as jn
import multiprocessing as mp
from typing import Tuple

"""
N= #particles
The None index indicates the event axis
File layout:
    - jet1:
        - 4mom: (None, N, 4) - 4 momenta stored in (e, px, py, pz) format for every particle.
        - coords: (None, N, 2) - (eta, phi) for every particle
        - features: (None, N, 9) - features for every particle: (eta, phi, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR)
        - mask: (None, N, 1) - mask (1 if particle is present, 0 otherwise) for each particle
    - jet2
        - 4mom: (None, N, 4) - 4 momenta stored in (e, px, py, pz) format for every particle.
        - coords: (None, N, 2) - (eta, phi) for every particle
        - features: (None, N, 9) - features for every particle: (eta, phi, log(pt), log(e), pt/pt_jet, e/e_jet, log(pt/pt_jet), log(e/e_jet), deltaR)
        - mask: (None, N, 1) - mask (1 if particle is present, 0 otherwise) for each particle
    - jet_features: (None, 7) - (tau1j1, tau2j1, tau3j1, tau1j2, tau2j2, tau3j2, mjj) for every event
    - jet_coords: (None, 2, 4) - (pt, eta, phi, m) for both jets
    - signal: (None) - signal bit (1 for signal, 0 otherwise)        
"""


R = 1.0
beta = 1.0

def process_jet(event:np.ndarray, jet:fj.PseudoJet, n_particles:int, nsub:jn.Nsubjettiness):
    idx = np.array([p.user_index() for p in jet.constituents()])
    particles = event[idx] #select the particles in this jet
    idx = np.argsort(particles[:,0]) #sort particles from low to high pt
    if(particles.shape[0]>n_particles): #get the top N particles according to pt
        idx = idx[-n_particles:]
    idx = idx[::-1] #reverse order (high to low)
    particles = particles[idx] #select these particles
    jet_particles = []
    for p in jet.constituents():
        jet_particles.extend([p.px(), p.py(), p.pz(), p.e()])
    return particles, (jet.pt(), jet.eta(), jet.phi(), jet.m()), jet.e(), nsub.getTau(3,jet_particles)

def prepare_data(events:np.ndarray, n_particles:int, jet_def:fj.JetDefinition, nsub:jn.Nsubjettiness, rotate_jets:bool=True):
    mom4 =      [np.zeros((events.shape[0], n_particles,4), dtype='float32'),
                 np.zeros((events.shape[0], n_particles,4), dtype='float32')]
    coords =    [np.zeros((events.shape[0], n_particles,2), dtype='float32'),
                 np.zeros((events.shape[0], n_particles,2), dtype='float32')]
    features =  [np.zeros((events.shape[0], n_particles,9), dtype='float32'),
                 np.zeros((events.shape[0], n_particles,9), dtype='float32')]
    mask =      [np.zeros((events.shape[0], n_particles,1),dtype='int8'),
                 np.zeros((events.shape[0], n_particles,1),dtype='int8')]
    jet_idx =   [np.zeros((events.shape[0], n_particles,1), dtype='int8'),
                 np.ones((events.shape[0], n_particles,1), dtype='int8')]
    feats = np.zeros((events.shape[0], 7)) #(mjj, tau1_j1, tau2_j1, tau3_j1, tau1_j2, tau2_j2, tau3_j2)
    jet_coords = np.zeros((events.shape[0], 2, 4)) #((pt_1, eta_1, phi_1, m_1), (pt_2, eta_2, phi_2, m_2))
    events = events.astype(np.float64) #in (pt, eta, phi) format
    for i in range(events.shape[0]):
        event = events[i]
        pjs = []
        for j in range(get_nparticles(event)):
            pj = fj.PseudoJet()
            pj.reset_PtYPhiM(event[j*3],event[j*3+1],event[j*3+2], 0.)
            pj.set_user_index(j)
            pjs.append(pj)
        jets = jet_def(pjs)
        idx2 = np.argsort([jet.pt() for jet in jets])[::-1]
        jet1, jet2 = jets[idx2[0]],jets[idx2[1]]
        #if(jet1.m() > jet2.m()):
        #    jet1, jet2 = jet2, jet1
        feats[i][-1] = (jet1+jet2).m()

        event = event.reshape((-1,3))
        for k, jet in enumerate((jet1,jet2)):
            coords3, jet_coordinates, jet_e, taus = process_jet(event, jet, n_particles, nsub)
            #coords3 = [(pt1, eta1, phi1), (pt2, eta2, phi2), ...]
            n = coords3.shape[0]

            delta_eta = coords3[:,1] - jet_coordinates[1]
            delta_phi = coords3[:,2] - jet_coordinates[2]

            #make sure delta phi is in between [-pi, pi]
            delta_phi[delta_phi<-np.pi] += 2*np.pi
            delta_phi[delta_phi> np.pi] -= 2*np.pi
            if(rotate_jets):
                #Note: coords3 is (pt, delta eta, delta phi)
                coords3[:,1] = delta_eta
                coords3[:,2] = delta_phi
            else:
                #Note: coords3 is (pt, eta, phi), (delta eta, delta phi) is stored in their respective variables
                #make sure phi is in between [-pi, pi]
                coords3[coords3[:,2]<-np.pi, 2] += 2*np.pi
                coords3[coords3[:,2]> np.pi, 2] -= 2*np.pi

            coords[k][i, :n,:] = coords3[:,1:] #eta/phi, rotated or unrotated
            #calculate 4mom
            e,px,py,pz = PtEtaPhi_to_EXYZ(coords3[:,0], coords3[:,1], coords3[:,2]) #use either rotated or unrotated coords
            mom4[k][i, :n,0] = e
            mom4[k][i, :n,1] = px
            mom4[k][i, :n,2] = py
            mom4[k][i, :n,3] = pz
            feats[i, k*3:k*3+3] = taus
            jet_coords[i,k,:] = jet_coordinates
            #calculate features
            #features = eta,phi, np.log(pt), np.log(e), rel_pt, rel_e, 
            #            np.log(rel_pt), np.log(rel_e), deltaR
            pt = coords3[:,0]
            rel_pt = pt/jet_coordinates[0]
            rel_e = e/jet_e
            deltaR = np.sqrt(np.square(delta_phi)+np.square(delta_eta))
            features[k][i,:n,:] = np.stack((delta_eta,delta_phi,
                                   np.log(pt),np.log(e),
                                   rel_pt, rel_e,
                                   np.log(rel_pt), np.log(rel_e), deltaR),axis=1)

            #calculate mask
            mask[k][i,:n,0] = (pt!=0)
    return [np.concatenate(mom4, axis=1), np.concatenate(coords, axis=1), np.concatenate(features, axis=1), np.concatenate(mask, axis=1), np.concatenate(jet_idx,axis=1)], feats, jet_coords

def setup_file(file:str, n_particles:int, max_size:int) -> h5py.File:
    fout = h5py.File(file, mode='w')
    fout.create_dataset("4mom",shape=(0, 2*n_particles, 4), dtype=np.float32, maxshape=(None,2*n_particles,4), chunks=True, compression='gzip')
    fout.create_dataset("coords",shape=(0, 2*n_particles, 2), dtype=np.float32, maxshape=(None,2*n_particles,2), chunks=True, compression='gzip')
    fout.create_dataset("features",shape=(0, 2*n_particles, 9), dtype=np.float32, maxshape=(None,2*n_particles,9), chunks=True, compression='gzip')
    fout.create_dataset("mask",shape=(0, 2*n_particles, 1), dtype=np.int8, maxshape=(None,2*n_particles,1), chunks=True, compression='gzip')
    fout.create_dataset("jet_idx",shape=(0, 2*n_particles, 1), dtype=np.int8, maxshape=(None,2*n_particles,1), chunks=True, compression='gzip')
    
    fout.create_dataset("jet_features", shape=(0,7), dtype=np.float32, maxshape=(None,7), chunks=True, compression='gzip')
    fout.create_dataset("jet_coords", shape=(0,2,4), dtype=np.float32, maxshape=(None,2,4), chunks=True, compression='gzip')
    fout.create_dataset("signal", shape=(0,), dtype=np.int8, maxshape=(None,), chunks=True, compression='gzip')
    return fout

def append_chunk(out_file:h5py.File, chunk):
    count = chunk[0].shape[0]
    for k in out_file.keys():
        out_file[k].resize(out_file[k].shape[0]+count, axis=0)
    out_file["4mom"][-count:,...] = chunk[0]
    out_file["coords"][-count:,...] = chunk[1]
    out_file["features"][-count:,...] = chunk[2]
    out_file["mask"][-count:,...] = chunk[3]
    out_file["jet_idx"][-count:,...] = chunk[4]
    out_file["jet_features"][-count:,...] = chunk[5]
    out_file["jet_coords"][-count:,...] = chunk[6]
    out_file["signal"][-count:] = chunk[7]

def writer(filename:str, out_queue:mp.Queue, n_particles:int, n_events:int, n_workers:int, sequential:bool=False):    
    outfile = setup_file(filename, n_particles, n_events)
    done_counter = 0
    if sequential:
        import queue, threading
        pq = queue.PriorityQueue()
        cond = threading.Condition()
        def append_chunk_thread():
            chunk_counter = 0
            while True:
                with cond:
                    cond.wait()
                while not pq.empty(): #keep checking the first item in the queue
                    idx, chunk = pq.get()
                    if chunk is None: return
                    if chunk_counter == idx:
                        print(f"Appending chunk {idx}!")
                        append_chunk(outfile, chunk)
                        chunk_counter += 1
                    else:
                        #put it back and wait for the next item to be added to the queue
                        pq.put((idx, chunk))
                        break
        t = threading.Thread(target=append_chunk_thread)
        t.daemon = True
        t.start()

        while done_counter < n_workers:
            res = out_queue.get()
            if(res == None):
                done_counter += 1
                print(f"Worker {done_counter} is done!")
                continue
            idx, chunk = res
            pq.put((idx, chunk))
            with cond:
                cond.notify()
        pq.put((2e15, None))
        with cond:
            cond.notify()
        t.join()
    else:
        while done_counter < n_workers:
            res = out_queue.get()
            if(res == None):
                done_counter += 1
                print(f"Worker {done_counter} is done!")
                continue
            idx, chunk= res
            print(f"Appending chunk {idx}")
            append_chunk(outfile, chunk)
    outfile.close()

def chunk_processor(infile:str, in_queue:mp.Queue, out_queue:mp.Queue, rotate_jets:bool=True, mjj_range=None, key:str='df'):
    print(f"Started worker {mp.current_process().name}")
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    nsub = jn.Nsubjettiness(R,beta,jn.Nsubjettiness.NormalizedMeasure, jn.Nsubjettiness.OnePass_KT_Axes)
    while (chunk := in_queue.get()) is not None:
        chunk_idx, (start, stop) = chunk
        print(f"Processing chunk {chunk_idx} ({start}:{stop})")
        df = pd.read_hdf(infile, key=key, start=start, stop=stop, mode='r')
        jet, feats, jet_coords = prepare_data(df.iloc[:,:2100].to_numpy(), n_particles, jet_def, nsub, rotate_jets=rotate_jets)
        if mjj_range is not None:
            idx = np.logical_and(feats[:,-1]>=mjj_range[0], feats[:,-1]<=mjj_range[1])
            out_queue.put((chunk_idx, (jet[0][idx], jet[1][idx], jet[2][idx], jet[3][idx], jet[4][idx], feats[idx], jet_coords[idx], df.iloc[:,2100].to_numpy()[idx])))
        else:
            out_queue.put((chunk_idx, (*jet, feats, jet_coords, df.iloc[:,2100].to_numpy())))
    out_queue.put(None)
    print(f"Finished worker {mp.current_process().name}")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("Preprocessing LHCO data")
    parser.add_argument("input", help="Input file (LHCO format)", type=str)
    parser.add_argument("output", help="Output file", type=str)
    parser.add_argument("--events", "-e", help="Number of events", type=int)
    parser.add_argument("--key", "-k", help="Key of the dataset in the input file", type=str, default='df')
    parser.add_argument("--offset", "-o", help="Skip this many events at the start of the file", default=0, type=int)
    parser.add_argument("--number", "-N", help="Number of particles", required=True, default=100, type=int)
    parser.add_argument("--n_proc", "-n", help="Number of processes to use", type=int, default=3)
    parser.add_argument("--no-rotate", help="Do *not* rotate the jets to (eta,phi)=(0,0)", action='store_true')
    parser.add_argument("--mjj", help="MJJ range. Format: lower:upper, in GeV", type=str, default=None)
    parser.add_argument("--chunksize", help="The chunksize", type=int, default=8192)
    parser.add_argument("--sequential", help="Ensure that the output is written sequentially", action='store_true')
    args = vars(parser.parse_args())
    
    if not os.path.exists(args["input"]):
        print("Input file does not exist")
        exit(-1)

    mjj_range = None
    if args["mjj"]:
        match = re.match(r"^(\d+):(\d+)$", args["mjj"])
        if(match):
            mjj_range = (float(match.group(1)),float(match.group(2)))
        else:
            print("Invalid mjj range")
            exit()
    infile = h5py.File(args["input"])
    try:
        shape = infile[f"{args['key']}/block0_values"].shape
    except KeyError as e:
        try:
            shape = infile[f"{args['key']}/table"].shape
        except KeyError:
            print("Could not extract size of dataset. Did you forget to specify a key?")
            exit()
    
    offset = args["offset"]
    if(args["events"] is None):
        n_events = shape[0]-offset
    else:
        n_events = min(args["events"],shape[0]-offset)
    n_particles = args["number"]

    infile.close()

    chunksize = args["chunksize"]
    chunks = [[i*chunksize+offset,min((i+1)*chunksize, n_events)+offset] for i in range((n_events+chunksize-1)//chunksize)]

    fname_split = os.path.splitext(args["output"])
    assert(fname_split[1]=='.h5')
    filenames = [fname_split[0]+f"-{i:d}.h5" for i in range(len(chunks))]

    if(args["n_proc"]>1):
        in_queue = mp.Queue()
        for i, chunk in enumerate(chunks): in_queue.put((i,chunk))
        for _ in range(args["n_proc"]): in_queue.put(None) #add the terminal signals
        out_queue = mp.Queue()

        processes = []
        for i in range(args["n_proc"]-1):
            process = mp.Process(target=chunk_processor, args=(args["input"], in_queue, out_queue, not args["no_rotate"], mjj_range, args["key"]))
            processes.append(process)
            process.start()

        process = mp.Process(target=writer, args=(args["output"], out_queue, n_particles, n_events, len(processes), args["sequential"]))
        processes.append(process)
        process.start()
        for p in processes:
            p.join()
        print("Done processing.")
    else:
        from tqdm import tqdm
        outfile = setup_file(args["output"], n_particles, n_events)
        jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
        nsub = jn.Nsubjettiness(R,beta,jn.Nsubjettiness.NormalizedMeasure, jn.Nsubjettiness.OnePass_KT_Axes)
        for chunk in tqdm(chunks):
            df = pd.read_hdf(args["input"], key=args["key"], start=chunk[0], stop=chunk[1], mode='r')
            jet, feats, jet_coords = prepare_data(df.iloc[:,:2100].to_numpy(), n_particles, jet_def, nsub, rotate_jets=not args["no_rotate"])
            if mjj_range is not None:
                idx = np.logical_and(feats[:,-1]>=mjj_range[0], feats[:,-1]<=mjj_range[1])
                append_chunk(outfile, (jet[0][idx], jet[1][idx], jet[2][idx], jet[3][idx], jet[4][idx], feats[idx], jet_coords[idx], df.iloc[:,2100].to_numpy()[idx]))
            else:
                append_chunk(outfile, (*jet, feats, jet_coords, df.iloc[:,2100].to_numpy()))
        outfile.close()
        print("Done processing.")