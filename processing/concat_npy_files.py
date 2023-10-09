import numpy as np
import h5py
import glob, os
import pandas as pd
from tqdm import tqdm
import tables
signal = 0

def openHDFfile(filename):
    if(os.path.isfile(filename)):
        os.remove(filename)
    file = pd.HDFStore(filename)
    colnames = [x for i in range(700) for x in [f"pt{i}", f"eta{i}", f"phi{i}"]]
    assert(len(colnames)==2100)
    if(signal is not None):
        colnames.append("signal")
    pd.DataFrame(dtype=np.float32, columns=colnames).to_hdf(file, "df",
            format="table", complevel=1,complib="blosc:zlib")
    return file, colnames

def writeToFile(events, file):
    if(signal is not None):
        assert columns[-1] == "signal"
        df = pd.DataFrame(events, dtype=np.float32, columns=columns[:-1])
        df["signal"] = signal
    else:
        df = pd.DataFrame(events, dtype=np.float32, columns=columns)
    df.to_hdf(file, "df", format="table", append="true",index=False, complevel=1, complib="blosc:zlib")
    assert not df.isnull().values.any(), "NaN in DF"

out_file = "events_DelphesPythia_v2_inneronly.h5"
ofile, columns = openHDFfile(out_file)
dirs = ["Output_DelphesPythia8_AnomalyDetection_v2_qcd_inneronly", "Output_DelphesPythia8_AnomalyDetection_v2_qcd_inneronly_batch2"]
for d in dirs:
    count = 0
    data = None
    for file in tqdm(glob.glob(f"{d}/*.npy")):
        npdata = np.load(file)[:,:2100]
        data = npdata if data is None else np.concatenate((data,npdata))
        count += npdata.shape[0]
        if(data.shape[0]>2000):
            assert not np.isnan(data).any(), "NaN encountered!"
            writeToFile(data, ofile)
            data = None
    if data is not None:
        writeToFile(data, ofile)
    print(f"{count=}")
ofile.close()

ofile = tables.open_file(out_file, 'r')
print(ofile.root.df.table.shape)