import h5py
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
from utils.coords import XYZ_to_PtEtaPhi
def prepare_data(data, N):
    data = data.to_numpy()
    events = data[:, :800]
    signal_bit = data[:,-1]
    particles = events.reshape((-1,events.shape[1]//4,4)) #E, px, py, pz

    #Calculate m
    coordssq = np.square(particles)
    m = np.sqrt(np.maximum(0,coordssq[:,:,0]-np.sum(coordssq[:,:,1:], axis=2))) #(B,N)
    mom4_jet = np.sum(particles, axis=1) #(B,4)
    m_jet = np.sqrt(np.maximum(0,mom4_jet[:,0]**2-np.sum(mom4_jet[:,1:]**2, axis=1))) #(B,)
    pt_jet = np.sqrt(mom4_jet[:,1]**2+mom4_jet[:,2]**2)

    #Select top N entries
    mom4 = particles[:,:N, :] #top N particles (B,N,4)
    mask = ~np.all(mom4==0, axis=2).reshape(-1,N,1).astype(np.int8) #(B,N,1)

    pt, eta, phi = XYZ_to_PtEtaPhi(mom4[:,:,1], mom4[:,:,2], mom4[:,:,3])
    phi -= np.pi #[-pi,pi]
    coords = np.stack((eta,phi),axis=2)
    deltaR = np.sqrt(np.square(phi)+np.square(eta))
    
    m = m[:,:N] #(B,N)
    pt = pt[:,:N] #(B,N)
    log_pt = np.log(pt+1e-30) #prevent nans
    log_pt[~mask] = 0 
    rel_pt = pt/(pt_jet.reshape(-1,1)+1e-30)
    rel_pt[~mask] = 0
    rel_m = m/(m_jet.reshape(-1,1)+1e-30)
    rel_m[~mask] = 0
    features = np.stack((eta, phi, pt, log_pt, rel_pt, m, rel_m, deltaR), axis=2) #(B,N,5)
    jet_coords = np.stack((pt_jet, m_jet), axis=1)
    is_bg = signal_bit==0
    is_sn = signal_bit==1
    return (mom4[is_bg], coords[is_bg], features[is_bg], mask[is_bg], jet_coords[is_bg], 0),(mom4[is_sn], coords[is_sn], features[is_sn], mask[is_sn], jet_coords[is_sn],1)

def chunk_processor(infile:pd.HDFStore, in_queue:mp.Queue, out_queue_bg:mp.Queue, out_queue_sn:mp.Queue):
    while (chunk := in_queue.get()) is not None:
        print(f"Processing chunk {chunk}")
        dataset = infile.select("table", start=chunk[0], stop=chunk[1])
        bg, sn = prepare_data(dataset, N=n_particles)
        out_queue_bg.put((chunk, bg))
        out_queue_sn.put((chunk, sn))
    # Signal the writer that this reader is done
    out_queue_bg.put(None)
    out_queue_sn.put(None)

def setup_file(outfile:h5py.File, n_events, n_particles):
    outfile.create_dataset("4mom", shape=(n_events, n_particles, 4), dtype='float32')
    outfile.create_dataset("coords", shape=(n_events, n_particles, 2), dtype='float32')
    outfile.create_dataset("features", shape=(n_events, n_particles, 8), dtype='float32')
    outfile.create_dataset("mask", shape=(n_events, n_particles, 1), dtype='int8')
    outfile.create_dataset("jet_coords", shape=(n_events, 2), dtype='float32')
    outfile.create_dataset("is_signal", shape=(n_events,), dtype='int8')

def writer(out_queue:mp.Queue, outfile:str, n_events:int, n_workers:int):
    done_counter = 0
    outfile = h5py.File(outfile, 'w')
    setup_file(outfile, n_events, n_particles)
    row_counter = 0
    while done_counter < n_workers:
        res = out_queue.get()
        if(res == None):
            done_counter += 1
            print(f"Worker {done_counter} is done!")
            continue
        chunk, (mom4, coords, feats, mask, jet_coords, is_signal) = res
        n = mom4.shape[0]
        print("Writing chunk", chunk)
        outfile["4mom"][row_counter:row_counter+n] = mom4
        outfile["coords"][row_counter:row_counter+n] = coords
        outfile["features"][row_counter:row_counter+n] = feats
        outfile["mask"][row_counter:row_counter+n] = mask
        outfile["jet_coords"][row_counter:row_counter+n] = jet_coords
        outfile["is_signal"][row_counter:row_counter+n] = is_signal
        row_counter += n
    outfile.close()


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("Preprocessing Top-Tagging data")
    parser.add_argument("input", help="Input file", type=str)
    parser.add_argument("output", help="Output file", type=str)
    parser.add_argument("--events", "-e", help="Number of events", type=int)
    parser.add_argument("--offset", "-o", help="Skip this many events at the start of the file", default=0, type=int)
    parser.add_argument("--number", "-N", help="Number of particles", default=-1, type=int)
    parser.add_argument("--n_proc", "-n", help="Number of processes to use", type=int, default=3)
    parser.add_argument("--chunksize", "-c", help="Chunk size", type=int, default=-1)
    args = vars(parser.parse_args())
    
    if not os.path.exists(args["input"]):
        print("Input file does not exist")
        exit(-1)

    store_in = pd.HDFStore(args["input"], 'r')
    storer = store_in.get_storer("table")
    shape = (storer.nrows, storer.ncols)
    
    offset = args["offset"]
    if(args["events"] is None):
        signalbit = store_in.select("table", start=offset, columns=["is_signal_new"])
        n_events = signalbit.shape[0]
        n_signal = (signalbit["is_signal_new"]==1).sum()
    else:
        n_events = min(args["events"],shape[0]-offset)
        signalbit = store_in.select("table", start=offset, stop=offset+n_events, columns=["is_signal_new"])
        n_signal = (signalbit["is_signal_new"]==1).sum()
    
    n_particles_in_file = 200
    n_particles = min(args["number"], n_particles_in_file) if args["number"]>0 else n_particles_in_file

    chunksize = args["chunksize"] if args["chunksize"]>0 else int(n_events/args["n_proc"])
    chunks = [[i*chunksize+offset,(i+1)*chunksize+offset] for i in range(n_events//chunksize+1)]
    chunks[-1][-1] = n_events+offset
    print("Computing chunks:", chunks)

    fname_split = os.path.splitext(args["output"])
    output_bg = fname_split[0]+"_bg.h5"
    output_sn = fname_split[0]+"_sn.h5"

    if(args["n_proc"]>2):
        in_queue = mp.Queue()
        for chunk in chunks: in_queue.put(chunk)
        for _ in range(args["n_proc"]): in_queue.put(None) #add the terminal signals
        out_queue_bg = mp.Queue()
        out_queue_sn = mp.Queue()

        processes = []
        for i in range(args["n_proc"]-2):
            process = mp.Process(target=chunk_processor, args=(store_in, in_queue, out_queue_bg, out_queue_sn))
            processes.append(process)
            process.start()

        process1 = mp.Process(target=writer, args=(out_queue_bg,output_bg, n_events-n_signal, len(processes)))
        process2 = mp.Process(target=writer, args=(out_queue_sn,output_sn, n_signal, len(processes)))
        processes.append(process1)
        processes.append(process2)
        process1.start()
        process2.start()
        for p in processes:
            p.join()
    else:
        import tqdm
        outfile_bg = h5py.File(output_bg, 'w')
        outfile_sn = h5py.File(output_sn, 'w')
        setup_file(outfile_bg, n_events-n_signal, n_particles)
        setup_file(outfile_sn, n_signal, n_particles)
        row_counter_bg, row_counter_sn = 0,0
        for chunk in tqdm.tqdm(chunks):
            dataset = store_in.select("table", start=chunk[0], stop=chunk[1])
            bg, sn = prepare_data(dataset, N=n_particles)
            n = bg[0].shape[0]
            outfile_bg["4mom"][row_counter_bg:row_counter_bg+n] = bg[0]
            outfile_bg["coords"][row_counter_bg:row_counter_bg+n] = bg[1]
            outfile_bg["features"][row_counter_bg:row_counter_bg+n] = bg[2]
            outfile_bg["mask"][row_counter_bg:row_counter_bg+n] = bg[3]
            outfile_bg["jet_coords"][row_counter_bg:row_counter_bg+n] = bg[4]
            outfile_bg["is_signal"][row_counter_bg:row_counter_bg+n] = bg[5]
            row_counter_bg += n

            n = sn[0].shape[0]
            outfile_sn["coords"][row_counter_sn:row_counter_sn+n] = sn[1]
            outfile_sn["features"][row_counter_sn:row_counter_sn+n] = sn[2]
            outfile_sn["mask"][row_counter_sn:row_counter_sn+n] = sn[3]
            outfile_sn["jet_coords"][row_counter_sn:row_counter_sn+n] = sn[4]
            outfile_sn["is_signal"][row_counter_sn:row_counter_sn+n] = sn[5]
            row_counter_sn += n
        outfile_bg.close()
        outfile_sn.close()
    store_in.close()