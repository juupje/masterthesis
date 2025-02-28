import h5py
import tqdm
from utils.calc import get_nparticles, create_chunks
from utils.coords import PtEtaPhi_to_EXYZ
import os, re
import numpy as np
import pandas as pd
import fastjet as fj
import pyjettiness as jn
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

def process_jet(event:np.ndarray, jet:fj.PseudoJet, n_particles:int, nsub:jn.Nsubjettiness) -> Tuple[np.ndarray, float, float, float]:
    idx = np.array([p.user_index() for p in jet.constituents()])
    particles = event[idx] #select the particles in this jet
    idx = np.argsort(particles[:,0]) #sort particles from low to high pt
    if(particles.shape[0]>n_particles): #get the top N particles according to pt
        idx = idx[-n_particles:]
    idx = idx[::-1] #reverse order (high to low)
    particles = particles[idx] #select these particles
    particles[:,1] -= jet.eta() #eta -> delta eta
    particles[:,2] -= jet.phi() #phi -> delta phi
    jet_particles = []
    for p in jet.constituents():
        jet_particles.extend([p.px(), p.py(), p.pz(), p.e()])
    return particles, (jet.pt(), jet.eta(), jet.phi(), jet.m()), jet.e(), nsub.getTau(3,jet_particles)


def prepare_data(events:np.ndarray, n_particles:int, jet_def:fj.JetDefinition, nsub:jn.Nsubjettiness):
    mom4 =      [np.zeros((events.shape[0], n_particles,4), dtype='float32'),
                 np.zeros((events.shape[0], n_particles,4), dtype='float32')]
    coords =    [np.zeros((events.shape[0], n_particles,2), dtype='float32'),
                 np.zeros((events.shape[0], n_particles,2), dtype='float32')]
    features =  [np.zeros((events.shape[0], n_particles,9), dtype='float32'),
                 np.zeros((events.shape[0], n_particles,9), dtype='float32')]
    mask =      [np.zeros((events.shape[0], n_particles,1),dtype='float32'),
                 np.zeros((events.shape[0], n_particles,1),dtype='float32')]
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
            n = coords3.shape[0]
            coords3[coords3[:,2]<-np.pi, 2] += 2*np.pi
            coords3[coords3[:,2]> np.pi, 2] -= 2*np.pi
            coords[k][i, :n,:] = coords3[:,1:] #eta/phi
            #calculate 4mom
            e,px,py,pz = PtEtaPhi_to_EXYZ(coords3[:,0], coords3[:,1], coords3[:,2])
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
            deltaR = np.sqrt(np.sum(np.square(coords3[:,1:]),axis=-1))
            features[k][i,:n,:] = np.stack((coords3[:,1],coords3[:,2],
                                   np.log(pt),np.log(e),
                                   rel_pt, rel_e,
                                   np.log(rel_pt), np.log(rel_e), deltaR),axis=1)

            #calculate mask
            mask[k][i,:n,0] = (pt!=0)
        
    return [[mom4[i], coords[i], features[i], mask[i]] for i in range(2)], feats, jet_coords

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

def setup_file(file:str, n_particles:int, size:int, set_max_shape_to_none=False) -> h5py.File:
    fout = h5py.File(file, mode='w')
    jets = [fout.create_group("jet1"),fout.create_group("jet2")]
    for jet in jets:
        for name,dim in zip(["4mom", "coords", "features", "mask"], [4,2,9,1]):
            jet.create_dataset(name,shape=(size,n_particles, dim), dtype=np.int8 if name=='mask' else np.float32, maxshape=(None,n_particles,dim) if set_max_shape_to_none else None,compression='gzip')
    fout.create_dataset("jet_features", shape=(size,7), dtype=np.float32, maxshape=(None,7) if set_max_shape_to_none else None, compression='gzip')
    fout.create_dataset("jet_coords", shape=(size,2,4), dtype=np.float32, maxshape=(None,2,4) if set_max_shape_to_none else None, compression='gzip')
    fout.create_dataset("signal", shape=(size,), dtype=np.int8, maxshape=(None,) if set_max_shape_to_none else None, compression='gzip')
    return fout

def process_chunk(infile:str, outfile:str, chunk:tuple, pos:int, mjj_range=None, key:str='df'):
    size = chunk[1]-chunk[0]
    fout = setup_file(outfile, n_particles, size)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    nsub = jn.Nsubjettiness(R,beta,jn.Nsubjettiness.NormalizedMeasure, jn.Nsubjettiness.OnePass_KT_Axes)
    tqdm_bar = tqdm.tqdm(total=size, position=pos)
    count = 0
    for piece in create_chunks(1024, total_size=size):
        df = pd.read_hdf(infile, key=key, start=piece[0]+chunk[0], stop=piece[1]+chunk[0], mode='r')
        (jet1, jet2), feats, jet_coords = prepare_data(df.iloc[:,:2100].to_numpy(), n_particles, jet_def, nsub)
        if mjj_range is not None:
            idx = np.logical_and(feats[:,-1]>=mjj_range[0], feats[:,-1]<=mjj_range[1])
        else:
            idx = np.ones(feats.shape[0], dtype=bool)
        n = np.sum(idx)
        for i, name in enumerate(["4mom", "coords", "features", "mask"]):
            fout[f"jet1/{name}"][count:count+n,...] = jet1[i][idx]
            fout[f"jet2/{name}"][count:count+n,...] = jet2[i][idx]
        fout["jet_features"][count:count+n,...] = feats[idx]
        fout["jet_coords"][count:count+n,...] = jet_coords[idx]
        fout["signal"][count:count+n] = df.iloc[:,2100].to_numpy()[idx]
        count += n
        tqdm_bar.update(feats.shape[0])
    tqdm_bar.close()
    fout.close()
    return count

if __name__=="__main__":
    import multiprocessing as mp
    import argparse
    parser = argparse.ArgumentParser("Preprocessing LHCO data")
    parser.add_argument("input", help="Input file (LHCO format)", type=str)
    parser.add_argument("output", help="Output file", type=str)
    parser.add_argument("--events", "-e", help="Number of events", type=int)
    parser.add_argument("--key", "-k", help="Key of the dataset in the input file", type=str, default='df')
    parser.add_argument("--offset", "-o", help="Skip this many events at the start of the file", default=0, type=int)
    parser.add_argument("--number", "-N", help="Number of particles", required=True, default=100, type=int)
    parser.add_argument("--append", "-a", help="Append to existing output file", action='store_true')
    parser.add_argument("--n_proc", "-n", help="Number of processes to use", type=int, default=3)
    parser.add_argument("--mjj", help="MJJ range. Format: lower:upper, in GeV", type=str, default=None)
    parser.add_argument("--split", help="Split the preprocessed dataset into subsets. Each split should match number:name", type=str,nargs='+')
    args = vars(parser.parse_args())
    
    if not os.path.exists(args["input"]):
        print("Input file does not exist")
        exit()
    if(args["append"] and not os.path.exists(args["output"])):
        print("Output file does not exist")
        exit()

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

    chunksize = int(n_events/args["n_proc"])
    chunks = [[i*chunksize+offset,(i+1)*chunksize+offset] for i in range(args["n_proc"])]
    chunks[-1][-1] = n_events+offset
    print("Computing chunks:", chunks)

    fname_split = os.path.splitext(args["output"])
    assert(fname_split[1]=='.h5')
    filenames = [fname_split[0]+f"-{i:d}.h5" for i in range(len(chunks))]

    def map_func(i):
        print("out file: " + filenames[i])
        return process_chunk(args["input"], filenames[i], chunks[i], pos=i, mjj_range=mjj_range, key=args["key"])

    with mp.Pool(args["n_proc"]) as p:
        counts = p.map(map_func, list(range(len(chunks))))

    print("Done processing.")
    print("Got counts: ", counts)
    count_total = np.sum(counts)

    print("Merging files:")
    if(args["append"]):
        outfile = h5py.File(args["output"], mode='a')
        try:
            check_file(outfile, n_particles)
        except AssertionError as e:
            print("Cannot append to file: "+str(e))
            exit()
        try:
            out_offset = extend_file(outfile, count_total)
        except Exception as e:
            print("Could not resize output file: " + str(e))
            exit()
    else:
        outfile = setup_file(args["output"], n_particles, count_total, set_max_shape_to_none=True)
        out_offset = 0

    for i in range(len(filenames)):
        subfile = h5py.File(filenames[i], mode='r')
        chunk = [out_offset, out_offset+counts[i]]
        print("Chunk:", str(chunk), "n_events:", counts[i])
        outfile["jet_features"][chunk[0]:chunk[1]] = subfile["jet_features"][:counts[i]]
        outfile["jet_coords"][chunk[0]:chunk[1]] = subfile["jet_coords"][:counts[i]]
        outfile["signal"][chunk[0]:chunk[1]] = subfile["signal"][:counts[i]]
        for name in ["4mom", "coords", "features", "mask"]:
            outfile[f"jet1/{name}"][chunk[0]:chunk[1],...] = subfile[f"jet1/{name}"][:counts[i]]
            outfile[f"jet2/{name}"][chunk[0]:chunk[1],...] = subfile[f"jet2/{name}"][:counts[i]]
        subfile.close()
        out_offset += counts[i]
        print(f"File {i+1:d} done")
    outfile.close()
    for file in filenames:
        os.remove(file)
    
    if(args["split"] and len(args["split"])>0):
        import split_dataset
        files = split_dataset.split_dataset(args["output"], args["output"], args["split"], shuffle=True)
        if(files):
            os.remove(args["output"])
        print(f"Preprocessed file {args['input']} and stored it to")
        for file in files:
            print(f"\t{file}")
    print("Done!")