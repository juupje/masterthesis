#! /usr/env/python3 
import os, sys
sys.path.append(os.getenv("HOME")+"/Analysis/lib")
from tqdm.contrib.concurrent import process_map
import numpy as np
import pandas as pd
import tables as pt
import matplotlib.pyplot as plt
import time

import jlogger as jl
import utils
logger = jl.JLogger()
NPROC = 4
DATA_DIR = "/scratch/work/geuskens/data"
PLOTS_DIR = os.getenv("HOME") + "/Analysis/plots"
files = {
    "LHCO_RnD": DATA_DIR + "/events_anomalydetection.h5", #reference
    #"MG pt900": DATA_DIR + "/MG_pt_900_events.h5", #first created
    #"MG pt800": DATA_DIR + "/MG_pt_800_events.h5", #to compare to MG_pt_900
    #"DP pt1000": DATA_DIR +"/DP_pt1000_events.h5", #compare to reference (v1)
    #"DP pt900": DATA_DIR +"/DP_pt900_events.h5", #compare to MG_pt_900
    #"DP pt500": DATA_DIR +"/DP_pt500_events.h5", #compare to reference (v2)
    "DP pt1000 new": DATA_DIR +"/DP_pt1000_events_new.h5", #compare to DP pt1000
    #"DP pt500 new": DATA_DIR +"/DP_pt500_events_new.h5", #compare to DP pt500
    "MG pt1000 mpi": DATA_DIR +"/MG_pt1000_events_mpi.h5", #compare to DP pt500
    "MG pt1000": DATA_DIR +"/MG_pt1000_events.h5", #compare to DP pt500
}

analysis = True
plot = True
maxEvents = 100_000 #We don't need more
bins = None

times = {}
#check if files exist
for key in files:
    if(not os.path.isfile(files[key])):
        print(f"File {files[key]} does not exist or is not a file.")
        exit()

for key in files:
    if(analysis):
        tic = time.time()
        file = pt.open_file(files[key], mode='r')
        if("df" in file.root):
            size = file.root.df.axis1.shape
        else:
            size = file.root.Particles.table.shape
        print(f"Processing file {files[key]} of shape {size}")
        file.close()

        def process_chunk(chunk):
            #print(f"Processing: {chunk}")
            n_particles = np.empty(chunk[1]-chunk[0], dtype=np.int32)
            df = pd.read_hdf(files[key],start=chunk[0], stop=chunk[1])
            i = 0
            for _, event in df.iterrows():
                    n_particles[i] = utils.get_nparticles(event)
                    i += 1
            return n_particles

        result = process_map(process_chunk, utils.create_chunks(chunksize=2048, total_size=min(maxEvents, size[0])), max_workers=NPROC)
        n_particles = np.concatenate(result)

        logger.store_log_data(dict(n_particles=n_particles), group_name=key, data_used=files[key],
            comment="'n_particles' contains the number of particles of the events")
        toc = time.time()
        times[key] = toc-tic

    if(plot):
        if(not analysis):
            print(f"Reading data for {key}")
            n_particles = logger.retrieve_logged_data("n_particles", group=key)

        print(f"Plotting {key}")
        if bins is None:
            bins = utils.calculate_bins(n_particles, 100)
        color = next(plt.gca()._get_lines.prop_cycler)["color"]
        n, bins, _ = plt.hist(n_particles, bins=bins, density=True, color=color, alpha=0.3, label=key, edgecolor=color)

if(plot):
    plt.xlabel("#particles per event")
    plt.ylabel("#events (normalized)")
    plt.legend()
    logger.log_figure(plt.gcf(), PLOTS_DIR+"/n_particles_mp.png", data_file="auto", comment="Histogram of number of particles per event",
        dpi=250)
for key in times:
    print(f"{key}: {times[key]:.4g}")