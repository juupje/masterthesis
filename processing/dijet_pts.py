import os, sys
sys.path.append(os.getenv("HOME")+"/Analysis/lib")
import numpy as np
import fastjet
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import jlogger as jl
import utils
import tables as pt
#import pyjettiness as jn

NPROC = 6
DATA_DIR = "/scratch/work/geuskens/data"

files = {
    "LHCO_RnD": DATA_DIR + "/events_anomalydetection_v2.h5", #reference
}

BS_SPLIT = 1_000_000 #first 1M events are bg
N_BACKGROUND = 10_000
N_SIGNAL = 10_000

logger = jl.JLogger("topk_pt")

R = 1.0
beta = 1.0
N_particles = 100
outqueue = mp.Queue()
inqueue = mp.Queue()
sentinel = None

def process_chunk(chunk):
    #with utils.quiet():
    df = pd.read_hdf(files[key], start=chunk[0], stop=chunk[1])
    if(df.shape[1]==2101):
        signal_bit = df.iloc[:,-1].to_numpy()
        df = df.iloc[:,:-1]
    else:
        signal_bit = np.zeros(df.shape[0])
    particles = df.to_numpy()
    #get top 100 particles:
    topk, idx = utils.get_topK(particles, N_particles, 3, 0)

    #print(idx.shape, df.shape)
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    jet_idx = np.zeros((topk.shape[0], N_particles))
    i = 0
    for _, event in df.iterrows():
        pjs = []
        for j in range(utils.get_nparticles(event)):
            pj = fastjet.PseudoJet()
            pj.reset_PtYPhiM(event[j*3],event[j*3+1],event[j*3+2], 0.)
            pj.set_user_index(j)
            pjs.append(pj)

        jets = jet_def(pjs)
        idx2 = np.argsort([jet.pt() for jet in jets])[::-1]
        jet1, jet2 = jets[idx2[0]],jets[idx2[1]]
        #!! why should the two leading jets be ordered according to their mass?
        #if(jet1.m() > jet2.m()):
        #    jet1, jet2 = jet2, jet1

        #well, this can't be vectorized :(
        idx_j1 = np.array([p.user_index() for p in jet1.constituents()])
        idx_j2 = np.array([p.user_index() for p in jet2.constituents()])

        jet_idx[i, np.isin(idx[i], idx_j1)] = 1
        jet_idx[i, np.isin(idx[i], idx_j2)] = 2
        i += 1

    topk = np.concatenate((topk, signal_bit.reshape(-1,1)), axis=1)
    return topk, jet_idx

def processor():
    for chunk in iter(inqueue.get, sentinel):
        topk, jet_idx = process_chunk(chunk)
        outqueue.put(("particles", chunk, topk))
        outqueue.put(("jet_indices", chunk, jet_idx))

def writer(n_chunks):
    fout = logger._open_file(mode='w')
    fout.create_dataset("particles", shape=(N_BACKGROUND+N_SIGNAL, N_particles*3+1))
    fout.create_dataset("jet_indices", shape=(N_BACKGROUND+N_SIGNAL, N_particles))
    pbar = tqdm(total=n_chunks)
    loc = {"particles": 0, "jet_indices":0}
    while True:
        output = outqueue.get()
        if(output is not None):
            name, chunk, arr = output
            assert(chunk[1]-chunk[0]==arr.shape[0])
            fout[name][loc[name]:loc[name]+arr.shape[0],...] = arr
            loc[name] += arr.shape[0]
            pbar.update(0.5)
        else:
            print("Done!")
            pbar.close()
            break
    print(fout["particles"][0])
    print(fout["particles"][-1])
    print(loc)
    fout.close()

for key in files:
    file = pt.open_file(files[key], mode='r')
    if("df" in file.root):
        size = file.root.df.axis1.shape
    else:
        size = file.root.Particles.table.shape
    file.close()

    print(f"Processing file {files[key]} of shape {size}")

    bg_chunks = utils.create_chunks(chunksize=1024, start=0, total_size=N_BACKGROUND)
    sn_chunks = utils.create_chunks(chunksize=1024, start=BS_SPLIT, total_size=N_SIGNAL)
    chunks = bg_chunks
    chunks.extend(sn_chunks)
    jobs = []
    proc = mp.Process(target=writer, args=(len(chunks),))
    proc.start()
    for i in range(NPROC):
        p = mp.Process(target=processor)
        jobs.append(p)
        p.start()
    
    for chunk in chunks:
        inqueue.put(chunk)
    for i in range(NPROC):
        inqueue.put(sentinel) #poison pill

    for p in jobs:
        p.join()
    print("Jobs joined!")
    outqueue.put(None) #poison pill
    proc.join()
    #process_map(process_chunk,
    #        chunks,
    #        max_workers=1)
    #print(fout["particles"][:,-1])
    #print(fout["particles"][0])
    #print(fout["particles"][-1])
    logger.log_data(DATA_DIR+"/processed/topk_pt.h5",
        comment="Top 100 particles by pT and their jets",
        data_used=files[key])