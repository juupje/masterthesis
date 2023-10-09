import argparse
import h5py
import re, os
import numpy as np

def perform_split(data_in:h5py.File, data_out, splits):
    def split(name, obj:h5py.HLObject, datasets_out, splits):
        if type(obj) is not h5py.Dataset: return
        dataset = np.array(obj, dtype=obj.dtype)
        for key in splits:
            d = datasets_out[key].create_dataset(name, data=dataset[splits[key],...])
            print(f"Created {name} for split {key}, shape: {d.shape}")
    data_in.visititems(lambda name,obj: split(name,obj, data_out, splits))

def split_dataset(infile:str, outfile:str, splits_str_list:list[str], start:int=0, end:int=-1, shuffle:bool=True, seed:int=None):
    print("Splitting dataset:")
    infile = h5py.File(infile, 'r')
    dataset = infile
    while (type(dataset) is not h5py.Dataset):
        for key in dataset.keys():
            dataset = dataset[key]
            break
    size = dataset.shape[0]
    if(end==-1): end = size
    elif(end > size):
        raise ValueError("`end` is greater than the size of the dataset")
    if(start>=end):
        raise ValueError("`start` is greater than `end`")
    assert start>=0, "`Start` is negative"
    
    splits = {}
    pat = re.compile(r"^(\d+(?:\.\d+)?):(\w+)$")
    total = 0
    for idx,split in enumerate(splits_str_list):
        split = split.strip()
        match = pat.match(split)
        if match:
            try:
                num = int(match.group(1))
            except ValueError:
                frac = float(match.group(1))
                num = int((end-start)*frac)
            splits[match.group(2)] = num
        else:
            match = re.match(r"^-1:(\w+)$", split)
            if(match is not None):
                if(idx==len(splits_str_list)-1):
                    num = int((end-start-total))
                    splits[match.group(1)] = num
                else:
                    raise ValueError("Split with size `-1` should be the last")
            else:
                raise ValueError("Invalid split:",split)
        total += num
    assert(total <= (end-start)), "Splits contain more data than there is in the dataset"
    idx = np.arange(start, end)
    rnd = np.random.default_rng(seed)
    if(shuffle):
        rnd.shuffle(idx)
    
    start,end = 0, 0
    data_out = {}
    files = [infile]
    ofile_split = os.path.splitext(outfile)
    for key in splits:
        end += splits[key]
        splits[key] = idx[start:end]
        out_file = h5py.File(ofile_split[0] + f"-{key}" + ofile_split[1], 'w')
        files.append(out_file)
        data_out[key] = out_file
        start = end
    perform_split(infile, data_out, splits)
    fnames = [file.filename for file in files[1:]]
    for file in files:
        print("Closing file", file)
        file.close()
    print("Split file into: ")
    for key in splits:
        print(key, f"size={splits[key].shape[0]:d}")
    return fnames
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a large dataset")
    parser.add_argument("input", help="Input file", type=str)
    parser.add_argument("output", help="Output file (will get suffixes)", type=str)
    parser.add_argument("--splits", "-s", nargs='+', required=True, type=str, help="Splits: provide as number:name, where `number` is the number of datapoints being added to this split")
    parser.add_argument("--start", "-b", help="Start point in the input dataset", default=0)
    parser.add_argument("--end", "-e", help="End point in the input dataset (only used when shuffling)", default=-1)
    parser.add_argument("--shuffle", help="Shuffle the dataset (or the start:end subset) before splitting", action='store_true')
    parser.add_argument("--seed", help="Shuffle seed", type=int)

    args = vars(parser.parse_args())
    try:
        split_dataset(args["input"], args["output"], args["splits"], args["start"], args["end"], args["shuffle"], args["seed"])
    except ValueError as e:
        print("Something went wrong:", e)
    print("Done!")