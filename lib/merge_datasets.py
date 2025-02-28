"""
Utility to merge datasets into one.

author: Joep Geuskens
"""

import argparse
import h5py
import re, os
import numpy as np

def perform_merge(files_in:list[str], data_out:h5py.File, slices:list, rnd:np.random.Generator):
    def to_slice(obj, size):
        if obj is None:
            return slice(0,size)
        if(type(obj) is float):
            obj = int(obj*size)
        if(type(obj) is int):
            if(obj>size): raise ValueError(f"Invalid slice: {obj} (bigger than size)")
            return rnd.choice(size,obj)
        if(type(obj) is tuple):
            return slice(obj[0],obj[1])
        else:
            raise ValueError(f"Invalid slice: {obj}")

    def copy(name:str, obj:h5py.HLObject, data_out:h5py.File, slice):
        if(type(obj) is not h5py.Dataset): return
        dataset = np.array(obj,dtype=obj.dtype)
        if(slice is not None): dataset[to_slice(slice)]
        data_out.create_dataset(name,data=dataset,maxshape=(None,*dataset.shape[1:]))
        print(f"Created {name}, shape: {dataset.shape}")

    def merge(name, obj:h5py.HLObject, dataset_out:h5py.File, slice):
        if type(obj) is not h5py.Dataset: return
        dataset = np.array(obj, dtype=obj.dtype)
        if(slice is not None): dataset = dataset[to_slice(slice)]
        if(name in data_out):
            size = dataset_out[name].shape[0]
            data_out[name].resize(size+dataset.shape[0], axis=0)
            data_out[name][size:,...] = dataset
            print(f"Extended {name}, shape: {data_out[name].shape}")
        else:
            print(f"ERROR: Unknown dataset {name}")
    print(files_in)
    first_file = h5py.File(files_in[0],'r')
    first_file.visititems(lambda name,obj: copy(name,obj,data_out,slices[0]))
    first_file.close()
    for idx, fname in enumerate(files_in[1:]):
        file = h5py.File(fname,'r')
        file.visititems(lambda name,obj: merge(name,obj, data_out, slices[idx]))
        file.close()

def merge_dataset(infiles:list[str], outfile:str, shuffle:bool=True, seed:int=None):
    print("Merging dataset:")
    data_out = h5py.File(outfile, 'w')
    
    slices = []
    in_files = []
    pat1 = re.compile(r"^(.*)(?:\@(\d+(?:\.\d+)?))$")
    pat2 = re.compile(r"^(.*)(?:\@(\d+):(\d+))$")
    for idx,split in enumerate(infiles):
        split = split.strip()
        match = pat1.match(split)
        if match:
            try:
                slices.append(int(match.group(2)))
            except ValueError:
                slices.append(float(match.group(2)))
            in_files.append(match.group(1))
        else:
            match = pat2.match(split)
            if match:
                begin, end = int(match.group(2)), int(match.group(3))
                slices.append(begin, end)
                in_files.append(match.group(1))
            else:
                slices.append(None)
                in_files.append(split)
    rnd = np.random.default_rng(seed)
    assert len(in_files)>1, "Needs more than 1 files to merge"
    perform_merge(in_files, data_out, slices, rnd)
    print("Merge done!")
    if(shuffle):
        def do_shuffle(name,obj):
            if type(obj) is not h5py.Dataset: return
            print("Shuffling", name)
            data_out[name][...] = rnd.shuffle(np.array(data_out[name]),axis=0)
        data_out.visititems(do_shuffle)

    print("Closing file", data_out)
    data_out.close()
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a large dataset")
    parser.add_argument("inputs", help="Input files", nargs='+', type=str)
    parser.add_argument("--output", "-o", help="Output file", required=True, type=str)
    parser.add_argument("--shuffle", help="Shuffle the dataset after merging", action='store_true')
    parser.add_argument("--seed", help="Shuffle seed", type=int)

    args = vars(parser.parse_args())
    try:
        merge_dataset(args["inputs"], args["output"], args["shuffle"], args["seed"])
    except ValueError as e:
        print("Something went wrong:", e)
    print("Done!")