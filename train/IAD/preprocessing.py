import sys
sys.path.insert(0,"../../lib")
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
import parallel_h5 as ph5
import multiprocessing as mp
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

def prepare_data(events:np.ndarray, n_particles:int, jet_def:fj.JetDefinition, nsub:jn.Nsubjettiness, rotate_jets:bool=True, reorder_by_mass:bool=False):
    mom4 =      [np.zeros((events.shape[0], n_particles,4), dtype='float32'),
                 np.zeros((events.shape[0], n_particles,4), dtype='float32')]
    coords =    [np.zeros((events.shape[0], n_particles,2), dtype='float32'),
                 np.zeros((events.shape[0], n_particles,2), dtype='float32')]
    features =  [np.zeros((events.shape[0], n_particles,9), dtype='float32'),
                 np.zeros((events.shape[0], n_particles,9), dtype='float32')]
    mask =      [np.zeros((events.shape[0], n_particles,1),dtype='int8'),
                 np.zeros((events.shape[0], n_particles,1),dtype='int8')]
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
        if reorder_by_mass:
            if(jet1.m() > jet2.m()):
                jet1, jet2 = jet2, jet1
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
        
    return [[mom4[i], coords[i], features[i], mask[i]] for i in range(2)], feats, jet_coords

'''
def check_file(file:h5py.File, n_particles:int) -> bool:
    assert all([(x in file) for x in ("jet1", "jet2", "jet_features", "jet_coords", "signal")]), "Output file has missing datasets"
    for jet in ["jet1", "jet2"]:
        assert all([(x in file) for x in (f"{jet}/4mom", f"{jet}/coords", f"{jet}/features", f"{jet}/mask")]), f"Output file has missing jet datasets ({jet})"
        for x, dim in zip(["4mom", "coords", "features", "mask"], (4,2,9,1)):
            shape = file[f"{jet}/{x}"].shape
            assert shape[1:]==(n_particles,dim), f"{jet}/{x} has the wrong shape: {shape} should be (None, {n_particles},{dim})"
    assert file["jet_features"].shape[1:]==(7,), f"jet_features has the wrong shape: {file['jet_features'].shape} should be (None, 7)"
    assert file["jet_coords"].shape[1:]==(2,4), f"jet_coords has the wrong shape: {file['jet_coords'].shape} should be (None, 2, 4)"
    assert len(file["signal"].shape[1:])==0, f"signal has the wrong shape: {file['signal'].shape} should be (None,)"
    return True

def extend_file(file:h5py.File, size:int) -> int:
    datasets = ["jet_features", "jet_coords", "signal"]
    for jet in ["jet1", "jet2"]:
        for name in ["4mom", "coords", "features", "mask"]:
            datasets.append(f"{jet}/{name}")
    sizes = [file[x].shape[0] for x in datasets]
    print(sizes)
    assert all([sizes[i]==sizes[0] for i in range(1,len(sizes))]), "Not all dataset sizes are equal!"
    for dataset in datasets:
        file[dataset].resize(size+sizes[0], axis=0)
    return sizes[0] #return the old size
'''

def append_chunk(out_file:h5py.File, chunk:ph5.ChunkData):
    count = chunk["jet1"]["mask"].shape[0]
    print(f"Appending chunk {chunk.get_index()} ({count}) on {mp.current_process().name}")
    def resize(name, obj:h5py.HLObject):
        if isinstance(obj, h5py.Dataset):
            obj.resize(obj.shape[0]+count, axis=0)
    out_file.visititems(resize)
    for jet in ["jet1", "jet2"]:
        out_file[f"{jet}/4mom"][-count:,...] = chunk["jet1"]["4mom"]
        out_file[f"{jet}/coords"][-count:,...] = chunk["jet1"]["coords"]
        out_file[f"{jet}/features"][-count:,...] = chunk["jet1"]["features"]
        out_file[f"{jet}/mask"][-count:,...] = chunk["jet1"]["mask"]
    out_file["jet_features"][-count:,...] = chunk["jet_features"]
    out_file["jet_coords"][-count:,...] = chunk["jet_coords"]
    out_file["signal"][-count:] = chunk["signal"]

def chunk_processor(chunkdata:ph5.ChunkData, infile:str, jet_def, nsub, rotate_jets:bool=True, mjj_range=None, reorder:bool=False, key:str='df'):
    chunk_idx, (start, stop) = chunkdata.get_index(), chunkdata.get_slice()
    print(f"Processing chunk {chunk_idx} ({start}:{stop}) on {mp.current_process().name}")
    df = pd.read_hdf(infile, key=key, start=start, stop=stop, mode='r')
    (jet1, jet2), feats, jet_coords = prepare_data(df.iloc[:,:2100].to_numpy(), n_particles, jet_def, nsub, rotate_jets=rotate_jets, reorder_by_mass=reorder)
    if mjj_range is not None:
        idx = np.logical_and(feats[:,-1]>=mjj_range[0], feats[:,-1]<=mjj_range[1])
    else:
        idx = slice()
    chunkdata["jet1"] = {"4mom": jet1[0][idx], "coords": jet1[1][idx], "features": jet1[2][idx], "mask": jet1[3][idx]}
    chunkdata["jet2"] = {"4mom": jet2[0][idx], "coords": jet2[1][idx], "features": jet2[2][idx], "mask": jet2[3][idx]}
    chunkdata["jet_features"] = feats[idx]
    chunkdata["jet_coords"] = jet_coords[idx]
    chunkdata["signal"] = df.iloc[:,2100].to_numpy()[idx]

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
    parser.add_argument("--reorder", help="Order jets by mass (lighter one first)", action='store_true')
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

    def setup_file(fout:h5py.File) -> h5py.File:
        jets = [fout.create_group("jet1"),fout.create_group("jet2")]
        for jet in jets:
            for name,dim in zip(["4mom", "coords", "features", "mask"], [4,2,9,1]):
                jet.create_dataset(name,shape=(0,n_particles, dim), dtype=np.int8 if name=='mask' else np.float32, maxshape=(None,n_particles,dim), chunks=True, compression='gzip')
        fout.create_dataset("jet_features", shape=(0,7), dtype=np.float32, maxshape=(None,7), chunks=True, compression='gzip')
        fout.create_dataset("jet_coords", shape=(0,2,4), dtype=np.float32, maxshape=(None,2,4), chunks=True, compression='gzip')
        fout.create_dataset("signal", shape=(0,), dtype=np.int8, maxshape=(None,), chunks=True, compression='gzip')
        return fout

    chunksize = args["chunksize"]
    chunks = [[i*chunksize+offset,min((i+1)*chunksize, n_events)+offset] for i in range((n_events+chunksize-1)//chunksize)]

    fname_split = os.path.splitext(args["output"])
    assert(fname_split[1]=='.h5')
    filenames = [fname_split[0]+f"-{i:d}.h5" for i in range(len(chunks))]
    
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    nsub = jn.Nsubjettiness(R,beta,jn.Nsubjettiness.NormalizedMeasure, jn.Nsubjettiness.OnePass_KT_Axes)
    if(args["n_proc"]>1):
        import parallel_h5 as ph5
        ph5.run_parallel(chunks, args["n_proc"]-1, append_chunk, setup_file,
                         args["output"], chunk_processor, chunk_processor_args=(args["input"], jet_def, nsub, not args["no_rotate"], mjj_range, args["reorder"], args["key"]), sequential=args["sequential"])
    else:
        from tqdm import tqdm
        outfile = setup_file(args["output"])
        for i, chunk in enumerate(tqdm(chunks)):
            chunkdata = ph5.ChunkData(i,chunk)
            chunk_processor(chunkdata, args["input"], jet_def, nsub, not args["no_rotate"], mjj_range, args["reorder"], args["key"])
            append_chunk(outfile, chunk)
        outfile.close()
        print("Done processing.")