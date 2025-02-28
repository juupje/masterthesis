"""
Utility library for parallel processing of HDF5 files.

You can use this library using the following pattern:
1. Define the chunks (usually as a list of tuples). For example, if your file has 1000 rows and you want to process 100 rows at a time, you can define the chunks as follows:
    `chunks = [(i, min(i+100, 1000)) for i in range(0, 1000, 100)]`
    so `chunks = [(0, 100), (100, 200), ..., (900, 1000)]`.
2. Define the `setup_file` function to setup the output file. This function takes only one argument: the output file.
    For example, you can create the groups/datasets in the output file using this function.
    ```
    def setup_file(outfile:h5py.File):
        outfile.create_group('totals')
        outfile.create_group('means')
        outfile.create_group('stds')
    ```
3. Define the `chunk_processor` function to process a chunk. This function takes two arguments: the chunk data and any additional arguments.
    It can use the `ChunkData.slice` attribute to get the slice of the chunk and retrieve the data from the input file accordingly.
    It can then process that data in whatever way is desired before writing the processed data back to ChunkData by modifying it in place as you would a dictionary.
    For example:
    ```
    def chunk_processor(chunkdata:ChunkData, input_file:h5py.File):
        data = input_file['data'][chunkdata.slice[0]:chunkdata.slice[1]]
        total, mean, std = np.sum(data,axis=-1), np.mean(data,axis=-1), np.std(data, axis=-1)
        chunkdata['total'] = total
        chunkdata['mean'] = mean
        chunkdata['std'] = std
        chunkdata['count'] = data.shape[0]
    ```
4. Define the `append_chunk` function to append a chunk to the output file. This function takes two arguments: the output file and the chunk data.
    This function should write the processed data in `chunkdata` to the output file.
    For example:
    ```
    def append_chunk(outfile:h5py.File, chunkdata:ChunkData):
        #first resize the datasets in the output file
        count = chunkdata['count']
        def resize(name, obj:h5py.HLObject):
            if isinstance(obj, h5py.Dataset):
                obj.resize(obj.shape[0]+count, axis=0)
        out_file.visititems(resize)
        outfile['totals'][-count:] = chunkdata['total']
        outfile['means'][-count:] = chunkdata['mean']
        outfile['stds'][-count:] = chunkdata['std']
    ```
5. Call the `run_parallel` function with the chunks, number of workers, `append_chunk`, `setup_file`, `chunk_processor`, and any additional arguments to pass to the `chunk_processor` function.
    For example:
    ```
    run_parallel(chunks, 4, append_chunk, setup_file, 'output.h5', chunk_processor, (input_file,))
    ```
author: Joep Geuskens
"""

import numpy as np
import h5py
import multiprocessing as mp
from typing import Callable, Tuple, List, Any

class ChunkData(dict):
    """
    Utility Class to store chunk data. Inherits from dict, so you can store any data you want in it.
    """
    def __init__(self, index:int, slice:Tuple, *args, **kwargs):
        self.slice = slice
        self.index = index

    def get_slice(self):
        return self.slice
    
    def get_index(self):
        return self.index

def writer(filename:str, out_queue:mp.Queue, n_workers:int, append_chunk:Callable[[h5py.File, ChunkData], None], setup_file:Callable[[h5py.File],None], sequential:bool=False):    
    """
    Takes `ChunkData` from `out_queue` and writes it to the output file using the `append_chunk` function.
    """
    outfile = h5py.File(filename, mode='w')
    setup_file(outfile)
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
            chunkdata:ChunkData = out_queue.get()
            if(chunkdata == None):
                done_counter += 1
                print(f"{done_counter} workers are done!")
                continue
            pq.put((chunkdata.get_index(), chunkdata))
            with cond:
                cond.notify()
        pq.put((2e15, None))
        with cond:
            cond.notify()
        t.join()
    else:
        while done_counter < n_workers:
            chunkdata = out_queue.get()
            if(chunkdata == None):
                done_counter += 1
                print(f"{done_counter} workers are done!")
                continue
            print(f"Appending chunk {chunkdata.get_index()}")
            append_chunk(outfile, chunkdata)
    outfile.close()

def run_parallel(chunks:List[Tuple], n_workers:int, append_chunk:Callable[[h5py.File, ChunkData], None], setup_file:Callable[[h5py.File],None],
                 filename:str, chunk_processor:Callable[[ChunkData,Any], None], chunk_processor_args:Tuple=(), sequential:bool=False):
    """
    Runs a parallel processing pipeline on a list of chunks. The chunks are processed by the chunk_processor function, which is called with the chunk data and the chunk_processor_args.
    The processed chunks are then written to the output file using the append_chunk function. The setup_file function is called before any chunks are written to the file.
    
    params
    ------
    chunks: List[Tuple]
        List of chunks to process. Each chunk can be represented as a tuple (usually of the form `(start, end)`).
    n_workers: int
        Number of worker processes to use. Since the operations are I/O bound, you shouldn't use more workers than the number of cores on your machine.
    append_chunk: Callable[[h5py.File, ChunkData], None]
        Function to append a chunk to the output file.
    setup_file: Callable[[h5py.File],None]
        Function to setup the output file. Called only once before any chunks are written.
    filename: str
        Output file name.
    chunk_processor: Callable[[ChunkData,Any], None]
        Function to process a chunk. The function should take a ChunkData object and any additional arguments.
        It should modify the ChunkData object in place.
    chunk_processor_args: Tuple
        Additional arguments to pass to the chunk_processor function.
    sequential: bool
        If True, the chunks are written to the file according to their order in `chunks`. This is useful if the order of the chunks is important.
    """
    in_queue = mp.Queue()
    for i, chunk in enumerate(chunks): in_queue.put(ChunkData(i,chunk))
    for _ in range(n_workers): in_queue.put(None)
    out_queue = mp.Queue()

    def worker(in_queue:mp.Queue, out_queue:mp.Queue):
        print(f"Started worker {mp.current_process().name}")
        while True:
            chunkdata = in_queue.get()
            if(chunkdata == None): break
            chunk_processor(chunkdata, *chunk_processor_args)
            out_queue.put(chunkdata)
        out_queue.put(None)
        print(f"Finished worker {mp.current_process().name}")

    processes = []
    for i in range(n_workers):
        p = mp.Process(target=worker, args=(in_queue, out_queue))
        processes.append(p)
        p.start()

    process = mp.Process(target=writer, args=(filename, out_queue, n_workers, append_chunk, setup_file, sequential))
    processes.append(process)
    process.start()
    for p in processes:
        p.join()
    print("Done processing!")