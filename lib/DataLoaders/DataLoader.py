import tensorflow as tf
import numpy as np
from time import time

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, bg, sn, batch_size:int, N_background:int, N_signal:int, features=None, seed:int=1, inputs:list=["coords", "features", "mask"]):
        self.inputs = inputs.copy()
        self.stopped=False
        self.rnd = np.random.default_rng(seed)
        N_data_bg = bg[inputs[0]].shape[0]
        N_data_sn = sn[inputs[0]].shape[0]
        tic = time()
        #do some checks
        assert(N_data_bg >= N_background and  N_data_sn >= N_signal), \
            "Dataset is smaller than number of events to be read from it"
        assert all([bg[s].shape == sn[s].shape for s in inputs]), \
            "Shapes of background and signal don't match"
        
        self.N = N_background+N_signal
        if(features is None):
            print(self.inputs)
            self.inputs.remove("features")
        ##### General remark: DO NOT use any kind of fancy slicing on h5 files, that is abysmally slow.
        ##### If needed, first load the entire file into memory as a numpy array (or a chunk of it) and index
        #determine which datapoints are going to be signals
        # we do this, so that we needn't shuffle later on!
        s_idx = np.zeros(self.N, dtype=bool) #default: all background
        s_idx[self.rnd.choice(self.N, N_signal, replace=False)] = True #these will be signals
        #initialize data
        self.data = {s:np.empty((self.N, *bg[s].shape[1:]), dtype=np.float32) for s in inputs}
        
        #select the required number of background events
        idx = np.sort(self.rnd.choice(N_data_bg, size=N_background, replace=False))
        for s in inputs: self.data[s][~s_idx] = np.array(bg[s])[idx] #put the chosen events into the background spots

        #select the required number of signal events
        idx = np.sort(self.rnd.choice(N_data_sn, size=N_signal, replace=False))
        for s in inputs: self.data[s][s_idx] = np.array(sn[s])[idx] #put the chosen events into the signal spots

        if(features != 'all' and features is not None):
            #select the required features
            self.data["features"] = self.data["features"][:,:,features] #yes, terrible code. I know
        self.labels = np.zeros((self.N, 2),dtype=int)
        self.labels[~s_idx,0] = 1 #background labels
        self.labels[s_idx,1] = 1 #signal labels
        self.N_batches = (self.N)//batch_size
        self.batch_size = batch_size
        self.time = time()-tic
        self.shuffle_time = 0
    
    def __len__(self):
        return self.N_batches

    def on_epoch_end(self):
        self.shuffle()
    
    def shuffle(self):
        tic = time()
        idx = np.arange(self.N)
        self.rnd.shuffle(idx)
        for s in self.inputs: self.data[s] = self.data[s][idx]
        self.labels = self.labels[idx]
        self.shuffle_time += time()-tic
        return idx

    def get_time_used(self):
        return self.time
    def get_time_used_shuffle(self):
        return self.shuffle_time

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        return [[self.data[s][start:end] for s in self.inputs], self.labels[start:end]]
    
    def stop(self):
        del self.data
        self.stopped=True

    def get_signal_ratio(self):
        n_signal = np.sum(self.labels[:,1])
        return n_signal/(self.labels.shape[0]-n_signal)