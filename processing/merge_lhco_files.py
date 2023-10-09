import h5py
import pandas as pd
import argparse
from tables import open_file
import numpy as np

def openHDFfile(filename, colnames):
    file = pd.HDFStore(filename, 'w')
    pd.DataFrame(dtype=np.float32, columns=colnames).to_hdf(file, "df",
            format="table", complevel=1,complib="blosc:zlib", append=False)
    return file

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LHCO HDF5 file merger")
    parser.add_argument("--input", "-i", help="Input files", nargs='+', type=str, required=True)
    parser.add_argument("--keys", "-k", help="Keys", nargs='+', type=str, default=['Particles'])
    parser.add_argument("--output", "-o", help="Ouput files", type=str, required=True)
    args = vars(parser.parse_args())

    if(len(args["keys"])==1):
        args["keys"] = args["keys"]*len(args["input"])
    elif(len(args["keys"])!=len(args["inputs"])):
        print("Number of keys does not match number of files")
        exit()
    if(len(args["input"])<2):
        print("Less than 2 files given.")
        exit()

    shapes = []
    colnames = []
    for idx,file in enumerate(args["input"]):
        data = open_file(file, mode='r')
        shapes.append(data.root.Particles.table.shape)
        colnames.append(data.root.Particles.table.colnames)
        data.close()
    if not all([shape[1:]==shapes[0][1:] for shape in shapes[1:]]):
        print("Not all datasets have the same shape")
        exit()
    if not all([cols==colnames[0] for cols in colnames[1:]]):
        print("Not all datasets have the same column names")
        for cols in colnames[1:]:
            if(cols != colnames[0]):
                print("Got:", cols)
                print("Expected:", colnames[0])
                exit()
        exit()
    
    hdffile = openHDFfile(args["output"], colnames[0])
    #print(hdffile['df'].shape)
    for idx,file in enumerate(args["input"]):
        print(f"Appending {file} - size: {shapes[idx][0]:d}")
        df = pd.read_hdf(file, args["keys"][idx], mode='r')
        print(df.shape)
        df.to_hdf(hdffile, "df",
            format="table", append="true",index=False, complevel=1, complib="blosc:zlib")
        print("Total size:", hdffile["df"].shape)
        del df
    hdffile.close()
