import h5py
import numpy as np
from utils.calc import create_chunks
#input files
DATA_DIR = os.path.join(os.getenv("/scratch/work/geuskens/data"),"/lhco/new/")
inputs = [DATA_DIR+"pt500/combined/N100-bg.h5", DATA_DIR+"pt500/combined/N100-bg2.h5", DATA_DIR+"pt500/combined/N100-bg3.h5"]
output = DATA_DIR+"pt500/N100-bg_fatjet-cut.h5"
cut = 850
chunksize = 100000

def copy_dataset(ofile:h5py.File, name:str, obj:h5py.HLObject, indices:np.ndarray, offset:int):
    if(type(obj) is not h5py.Dataset): return
    count = offset
    print(name)
    for chunk in create_chunks(chunksize, start=0, total_size=obj.shape[0]):
        print(chunk)
        data = np.array(obj[chunk[0]:chunk[1],...])
        data = data[indices[chunk[0]:chunk[1]]]
        ofile[name][count:count+data.shape[0],...] = data
        count += data.shape[0]
    print(f"Copied {name}: {obj.shape} -> ({offset:d}:{count:d}, {', '.join((str(x) for x in obj.shape[1:]))})")

ofile = h5py.File(output, 'w')
#first get the event indices
indices = []
shapes = {}
for inp in inputs:
    ifile = h5py.File(inp, 'r')
    pt = np.array(ifile["jet_coords"][:,:,0])
    max_pt = np.max(pt, axis=1)
    selected = max_pt>cut
    indices.append(selected)
    ifile.close()
total = sum([np.sum(l) for l in indices])
print("Total size: ", total)
count = 0

def create(name, obj):
    if(type(obj) is h5py.Dataset):
        ofile.create_dataset(name,shape=(total,*obj.shape[1:]))
        print("Created dataset", name)

for i,inp in enumerate(inputs):
    print("Processing file: ", inp)
    print(" selected events: ", np.sum(indices[i]))
    ifile = h5py.File(inp, 'r')
    if(i==0):
        print("Creating datasets...")
        ifile.visititems(create)
    print("Starting copying...")
    def _copy(name, obj):
        copy_dataset(ofile, name, obj, indices=indices[i], offset=count)
    ifile.visititems(_copy)
    count += np.sum(indices[i])
    ifile.close()
ofile.close()
