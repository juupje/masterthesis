import os, sys
sys.path.append(os.getenv("HOME")+"/Analysis/lib")
import numpy as np
import fastjet
import pandas as pd
from tqdm.contrib.concurrent import process_map
import jlogger as jl
import utils
import tables as pt
import pyjettiness as jn

NPROC = 6
DATA_DIR = "/scratch/work/geuskens/data"
signal=True
files = {
    "LHCO_RnD": DATA_DIR + "/events_anomalydetection_v2.h5", #reference
    #"DP pt1000 R0.8": DATA_DIR +"/DP_pt1000_events.h5", #DP run with R_fat=0.8 and no mpi
    #"DP pt1000": DATA_DIR +"/DP_pt1000_events_new.h5", #DP run with R_fat=1.0 and no mpi
    #"MG pt1000": DATA_DIR +"/MG_pt1000_events.h5", #MG run with R_fat=1.0 and no mpi
}

# output format:
# 'pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2'
maxEvents = np.infty
logger = jl.JLogger("features")

R = 1.0
beta = 1.0
Ntau = 3

def process_chunk(chunk, signal=signal):
    #with utils.quiet():
    df = pd.read_hdf(files[key], start=chunk[0], stop=chunk[1])
    if(signal):
        if(df.shape[1]==2101):
            df = df.loc[df[2100]==1]
        else:
            return np.empty((0,(4+Ntau)*2+2), dtype=np.float32)
    else:
        if(df.shape[1]==2101):
            df = df.loc[df[2100]==0]
    features = np.empty((df.shape[0],(4+Ntau)*2+1), dtype=np.float32)
    i = 0
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    nsub = jn.Nsubjettiness(R,beta,jn.Nsubjettiness.NormalizedMeasure, jn.Nsubjettiness.OnePass_KT_Axes)
    
    for _, event in df.iterrows():
        pjs = []
        for j in range(utils.get_nparticles(event)):
            pj = fastjet.PseudoJet()
            pj.reset_PtYPhiM(event[j*3],event[j*3+1],event[j*3+2], 0.)
            pjs.append(pj)

        #cluster = fastjet.ClusterSequence(pjs, jet_def)
        #jets = fastjet.sorted_by_pt(cluster.inclusive_jets(30))
        jets = jet_def(pjs)
        idx = np.argsort([jet.pt() for jet in jets])[::-1]
        jet1, jet2 = jets[idx[0]],jets[idx[1]]
        #!! why should the two leading jets be ordered according to their mass?
        #if(jet1.m() > jet2.m()):
        #    jet1, jet2 = jet2, jet1

        del idx,jets
        jet1_particles, jet2_particles = [], []
        for p in jet1.constituents():
            jet1_particles.extend([p.px(), p.py(), p.pz(), p.e()])
        for p in jet2.constituents():
            jet2_particles.extend([p.px(), p.py(), p.pz(), p.e()])

        features[i,:] = [jet1.px(),jet1.py(),jet1.pz(),jet1.m(),
                        *nsub.getTau(Ntau, jet1_particles),    
                        jet2.px(),jet2.py(),jet2.pz(),jet2.m(),
                        *nsub.getTau(Ntau, jet2_particles),
                        (jet1+jet2).m()]
        i += 1
    return features

for key in files:
    store_key = key +(" S" if signal else "")
    if(logger.exists_data(store_key)):
        if(not utils.prompt_overwrite(store_key)): continue
    
    file = pt.open_file(files[key], mode='r')
    if("df" in file.root):
        size = file.root.df.axis1.shape
    else:
        size = file.root.Particles.table.shape
    file.close()

    print(f"Processing file {files[key]} of shape {size}")
    result = process_map(process_chunk,
            utils.create_chunks(chunksize=1024, total_size=min(maxEvents, size[0])),
            max_workers=NPROC)
    result = np.concatenate(result)
    logger.store_log_data({"features": result}, group_name=store_key,
        comment="Two leading pT jets and invariant dijet mass" + (" - signal" if signal else ""),
        data_used=files[key])