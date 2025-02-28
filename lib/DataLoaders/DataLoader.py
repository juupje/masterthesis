import keras
import numpy as np
from time import time
from utils.shuffling import do_the_shuffle
from typing import List

class BaseDataLoader(keras.utils.PyDataset):
    """
    Abstract class for a data loader.
    """
    def __init__(self, seed=0, workers=1, use_multiprocessing=False, **kwargs):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, **kwargs)
        self.stopped=False
        self.rnd = np.random.default_rng(seed)
        self.tic = time()
        self.shuffle_time = 0
        self.data, self.labels, self.inputs = None, None, None

    def finish_init(self, batch_size, N_batches, N):
        self.time = time()-self.tic
        self.batch_size = batch_size
        self.N_batches = N_batches
        self.N = N
        assert hasattr(self, "labels") and self.labels is not None, "No labels have been set!"
        assert hasattr(self, "data") and self.data is not None, "No data has been set!"
        assert hasattr(self, "inputs") and self.inputs is not None, "No inputs have been set!"

    def get_signal_ratio(self):
        """
        NOTE: this returns the ratio of labels that are equal to 1.
        For the IAD, this is thus the ratio of SR events to total events!
        """
        n_signal = np.sum(self.labels[:,1])
        return n_signal/(self.labels.shape[0]-n_signal)

    def __len__(self):
        return self.N_batches

    def on_epoch_end(self):
        self.shuffle()
    
    def shuffle(self):
        tic = time()
        idx = np.arange(self.N)
        self.rnd.shuffle(idx)
        do_the_shuffle(self.data, idx)
        do_the_shuffle(self.labels, idx)
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
        return tuple(self.data[s][start:end] for s in self.inputs), self.labels[start:end]
    
    def stop(self):
        if hasattr(self, "data"):
            del self.data
        if hasattr(self, "labels"):
            del self.labels
        self.stopped=True

class DataLoader(BaseDataLoader):
    def __init__(self, bg, sn, batch_size:int, N_background:int, N_signal:int, features:List[int]=None, seed:int=1, inputs:list=["coords", "features", "mask"]):
        """
        Takes `N_background` background events and `N_signal` signal events from the given data.
        The data is shuffled.

        params
        ------
        bg: dict
            Dictionary containing the background data
        sn: dict
            Dictionary containing the signal data
        batch_size: int
            The batch size
        N_background: int
            The number of background events to be used
        N_signal: int
            The number of signal events to be used
        features: list
            The features to be used. If None, all features are used. If 'all', all features are used. Otherwise, the integers specify the indices of the features to be used. 
        seed: int
            The seed for the random number generator
        inputs: list
            The inputs to be used. Default: ["coords", "features", "mask"]
            If want to select features, you have to include "features" in the list.
            All inputs should be present as keys in `bg` and `sn`.
        """
        super().__init__(seed=seed)
        self.inputs = inputs.copy()
        N_data_bg = bg[inputs[0]].shape[0]
        N_data_sn = sn[inputs[0]].shape[0]
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
        self.finish_init(batch_size, self.N//batch_size)