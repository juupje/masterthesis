import numpy as np
import tensorflow as tf
from .Discriminator import Discriminator
from time import time
from utils import shuffling

class RegressionDataLoader(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size:int, N_data, data_slice=None, features=None, seed:int=1, regression_feature:int=-1,
                 inputs:list=["coords", "features", "mask"], particles:int=30, njets:int=2,discriminator:Discriminator=None,
                 log:bool=True, shift_mean:bool=True):
        self.inputs = inputs.copy()
        self.stopped = False
        self.rnd = np.random.default_rng(seed)
        self.njets = njets
        self.jets = [f"jet{idx}" for idx in range(1,1+njets)]
        data_slice = data_slice or slice(None,None,None)
        N_data_in_dataset = data[self.jets[0]+"/"+inputs[0]].shape[0]
        self.log = log
        if(features is None):
            self.inputs.remove("features")

        tic = time()
        #apply the discriminator
        disc = discriminator if discriminator is not None else Discriminator("jet_features/-1", None, None)
        idx = np.where(disc.apply(data, data_slice))[0]
        N_data_in_dataset = len(idx)
        if(discriminator):
            print(f"{N_data_in_dataset} events passed the discriminator")

        #find background and signal events
        assert N_data_in_dataset>=N_data, "Not enough events left after the discriminator"
        if(N_data < 0): N_data = N_data_in_dataset
        self.N = N_data

        #select the required number of background events
        idx = np.sort(self.rnd.choice(idx, size=self.N, replace=False))
        self.data = {jet: {s: np.array(data[f"{jet}/{s}"][data_slice, :particles])[idx] for s in inputs} for jet in self.jets}
        self.target = np.array(data["jet_features"][data_slice,regression_feature])[idx]
        if(self.log):
            self.target = np.log10(self.target)
        if(shift_mean):
            self.shift = np.mean(self.target)
            self.target -= self.shift
        else:
            self.shift = 0

        if(features != 'all' and features is not None):
            #select the required features
            for jet in self.jets: self.data[jet]["features"] = self.data[jet]["features"][:,:,features] #yes, terrible code. I know
        self.N_batches = (self.N)//batch_size
        self.batch_size = batch_size
        self.time = time()-tic
        self.shuffle_time = 0
    
    def shuffle(self):
        tic = time()
        idx = np.arange(self.N)
        self.rnd.shuffle(idx)
        shuffling.do_the_shuffle(self.data, idx)
        shuffling.do_the_shuffle(self.target, idx)
        self.shuffle_time += time()-tic
        return idx

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        return [[self.data[jet][s][start:end] for jet in self.jets for s in self.inputs], self.target[start:end]]

    def __len__(self):
        return self.N_batches

    def on_epoch_end(self):
        self.shuffle()

    def get_time_used(self):
        return self.time
    def get_time_used_shuffle(self):
        return self.shuffle_time
    
    def stop(self):
        del self.data
        self.stopped=True

    def get_shift(self):
        return self.shift
    
    def do_back_trafo(self, pred):
        x = pred+self.shift
        return np.power(10, x) if self.log else x