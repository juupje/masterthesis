import pandas as pd
import numpy as np
import jlogger as jl
import h5py
import random

DATA_DIR = os.getenv("DATA_DIR")

output = DATA_DIR + "/events_anomalydetection_v2_SR.h5"
N_background = 100_000
N_signal = 65_000
N_test = 20_000 #half signal, half background
val_split = 0.2
mjj_min = 3300
mjj_max = 3700

N = 4+3 #px,py,pz,m and tau1..3
PX = lambda x: x*N+0 #x is 0 or 1
PY = lambda x: x*N+1
PZ = lambda x: x*N+2
M = lambda x: x*N+3

features = pd.read_hdf(DATA_DIR+"/events_features_v2.h5") #small enough to fit in memory at once
logger = jl.JLogger()

def m(x1,y1,z1,m1,x2,y2,z2,m2):
    p1_2 = np.square(x1)+np.square(y1)+np.square(z1)
    p2_2 = np.square(x2)+np.square(y2)+np.square(z2)
    E1_2 = np.square(m1)+p1_2
    E2_2 = np.square(m2)+p2_2
    return np.sqrt(E1_2+E2_2+2*np.sqrt(E1_2)*np.sqrt(E2_2)-p1_2-p2_2-2*(x1*x2+y1*y2+z1*z2))

def extract(start, stop, features):
    data = pd.read_hdf(DATA_DIR+"/events_anomalydetection_v2.h5", start=start, stop=stop)
    fts = features.iloc[start:stop].to_numpy()
    mjj = m(fts[:,PX(0)], fts[:,PY(0)], fts[:,PZ(0)],fts[:,M(0)], fts[:,PX(1)], fts[:,PY(1)], fts[:,PZ(1)], fts[:,M(1)])
    idx = np.logical_and(mjj>=mjj_min, mjj<=mjj_max)
    selected = data.iloc[idx].to_numpy()
    idx_bg = selected[:,-1]==0
    return selected[idx_bg], selected[~idx_bg]

bg_train_count = 0
bg_val_count = 0
bg_test_count = 0
sn_train_count = 0
sn_val_count = 0
sn_test_count = 0
val_bg = int(N_background*val_split)
val_sn = int(N_signal*val_split)
train_bg = N_background-val_bg
train_sn = N_signal-val_sn

hdf5_file = h5py.File(output, mode='w')
hdf5_file.create_dataset('x_train_bg', (train_bg,2100), np.float32, compression="gzip")
hdf5_file.create_dataset('x_val_bg', (val_bg,2100), np.float32, compression="gzip")
hdf5_file.create_dataset('x_train_sn', (train_sn,2100), np.float32, compression="gzip")
hdf5_file.create_dataset('x_val_sn', (val_sn,2100), np.float32, compression="gzip")

hdf5_file_train = h5py.File(output.replace(".h5", "-test.h5"), mode='w')
hdf5_file_train.create_dataset('x_test_bg', (N_test//2,2100), np.float32, compression="gzip")
hdf5_file_train.create_dataset('x_test_sn', (N_test//2,2100), np.float32, compression="gzip")

i = 0
max_iter = 1_000_000//4000 #max number of background
while(bg_train_count<train_bg or bg_val_count < val_bg or bg_test_count < N_test//2):
    print(f"BG: {i}/{max_iter:d} {bg_train_count}/{train_bg} {bg_val_count}/{val_bg} {bg_test_count}/{N_test//2}")
    if(i > max_iter):
        print("Ran out of data. Stopping...")
        break
    bg_i, _ = extract(i*4000, (i+1)*4000, features)
    if(bg_train_count < train_bg):
        n = min(bg_train_count+bg_i.shape[0], train_bg)-bg_train_count
        hdf5_file["x_train_bg"][bg_train_count:bg_train_count+n,...] = bg_i[:n,:-1]
        bg_train_count += n
    elif(bg_val_count < val_bg):
        n = min(bg_val_count+bg_i.shape[0], val_bg)-bg_val_count
        hdf5_file["x_val_bg"][bg_val_count:bg_val_count+n,...] = bg_i[:n,:-1]
        bg_val_count += n
    else:
        n = min(bg_test_count+bg_i.shape[0], N_test//2)-bg_test_count
        hdf5_file_train["x_test_bg"][bg_test_count:bg_test_count+n,...] = bg_i[:n,:-1]
        bg_test_count += n
    i += 1

i = 0
max_iter = 100_000//4000 #max number of background
while(sn_train_count < train_sn or sn_val_count < val_sn or sn_test_count < N_test//2):
    print(f"SN: {i}/{max_iter:d} {sn_train_count}/{train_sn} {sn_val_count}/{val_sn} {sn_test_count}/{N_test//2}")
    if(i > max_iter):
        print("Ran out of data. Stopping...")
        break
    _, sn_i = extract(1_000_000+i*4000, 1_000_000+(i+1)*4000, features)
    n_left = sn_i.shape[0]
    n_train = min(sn_train_count+n_left, train_sn)-sn_train_count
    n_left -= n_train
    n_val = min(sn_val_count+n_left, val_sn)-sn_val_count
    n_left -= n_val
    n_test = min(sn_test_count+n_left, N_test//2)-sn_test_count

    if(n_train>0):
        hdf5_file["x_train_sn"][sn_train_count:sn_train_count+n_train,...] = sn_i[:n_train,:-1]
        sn_train_count += n_train
    if(n_val>0):
        hdf5_file["x_val_sn"][sn_val_count:sn_val_count+n_val,...] = sn_i[n_train:n_train+n_val,:-1]
        sn_val_count += n_val
    if(n_test>0):
        hdf5_file_train["x_test_sn"][sn_test_count:sn_test_count+n_test,...] = sn_i[n_train+n_val:n_train+n_val+n_test,:-1]
        sn_test_count += n_test
    i += 1

hdf5_file.close()
logger.log_data(output, "x_train_bg,x_val_bg,x_train_sn,x_val_sn",
                comment="A subset of background and signal events in the signal region, stored in seperate datasets",
                data_used="events_anomalydetection_v2.h5:events_features_v2.h5")
logger.log_data(output.replace(".h5", "-test.h5"), "x_test_bg,x_test_sn",
                comment="A test subset of background and signal events in the signal region, stored in seperate datasets",
                data_used="events_anomalydetection_v2.h5:events_features_v2.h5")
