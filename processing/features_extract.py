import numpy as np
import fastjet
import pandas as pd
from tqdm.contrib.concurrent import process_map
import utils
import pyjettiness as jn
import h5py

# output format:
# 'pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2'

R = 1.0
beta = 1.0
Ntau = 3

def process_chunk(chunk:tuple[int,int], fname:str, signal:bool) -> np.ndarray:
    #with utils.quiet():
    df = pd.read_hdf(fname, start=chunk[0], stop=chunk[1])
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
        for j in range(utils.calc.get_nparticles(event)):
            pj = fastjet.PseudoJet()
            pj.reset_PtYPhiM(event[j*3],event[j*3+1],event[j*3+2], 0.)
            pjs.append(pj)

        jets = jet_def(pjs)
        idx = np.argsort([jet.pt() for jet in jets])[::-1]
        jet1, jet2 = jets[idx[0]],jets[idx[1]]
        if(jet1.m() > jet2.m()):
            jet1, jet2 = jet2, jet1

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

if __name__ == "__main__":
    import argparse
    import os
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", help="Input file")
    argparser.add_argument("output", help="Output file")
    argparser.add_argument("--key", help="Key to use in the input file", default="df")
    argparser.add_argument("--signal", "-s", help="Signal or background", action="store_true")
    argparser.add_argument("--nprocs", "-n", type=int, help="Number of processes to use", default=1)
    argparser.add_argument("--chunksize", "-c", type=int, help="Size of chunks to process", default=1024)
    argparser.add_argument("--events", "-N", type=int, help="Number of events to process")
    argparser.add_argument("--offset", type=int, help="Number of events to offset")
    argparser.add_argument("--format", "-f", help="Output format", choices=['pandas', 'h5'], default="pandas")
    args = vars(argparser.parse_args())

    if not os.path.isfile(args["input"]):
        print(f"File {args['input']} not found")
        exit(1)

    file = h5py.File(args["input"], mode='r')
    if(args["key"] in file):
        size = file[f"{args['key']}/axis1"].shape[0]
    file.close()
    start = args["offset"] if args["offset"] else 0
    events = min(size-start, args["events"] if args["events"] else size)
    
    print("Size:", size, "Start:", start, "events:", events)

    result = None
    if args["nprocs"] > 1:
        print(f"Processing file {args['input']} of shape {size} using {args['nprocs']} processes")
        def func(chunk):
            return process_chunk(chunk, args["input"], signal=args["signal"])
        results = process_map(func,
                utils.calc.create_chunks(chunksize=args["chunksize"], start=start, total_size=events),
                max_workers=args["nprocs"])
        result = np.concatenate(results, axis=0)
    else:
        print(f"Processing file {args['input']} of shape {size}")
        results = []
        for chunk in utils.calc.create_chunks(chunksize=args["chunksize"], start=start, total_size=events):
            print("Processing chunk", chunk)
            results.append(process_chunk(chunk, args["input"], signal=args["signal"]))
        result = np.concatenate(results, axis=0)

    if args["format"] == "pandas":
        result = pd.DataFrame(result)
        result.to_hdf(args["output"], key=args["key"], mode="w")
    else:
        with h5py.File(args["output"], "w") as f:
            f.create_dataset("data", data=result)