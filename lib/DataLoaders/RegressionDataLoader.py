import numpy as np
import keras
from .Discriminator import Discriminator
from time import time
from utils import shuffling
from typing import Dict

class RegressionDataLoader(keras.utils.PyDataset):
    """
    Loads a dataset for regression tasks.

    - Specifically for multi-jet datasets
    - Supports option to first take a slice from the data before selecting the events according to `N_data`
        Useful for splitting the data into training and validation sets.
    - Support discriminator
    - Does not support decoder inputs
    - Does not support oversampling
    - Does not support adding noise to the features
    """
    def __init__(self, data:Dict[np.ndarray], batch_size:int, N_data:int, data_slice=None, features=None, seed:int=1, regression_feature:int=-1,
                 inputs:list=["coords", "features", "mask"], particles:int=30, njets:int=2,discriminator:Discriminator=None,
                 log:bool=True, shift_mean:bool=True):
        """
        params
        ------
        data: dict,
            the data dictionary
        batch_size: int,
            Size of each batch.
        N_data: int,
            Number of events to load.
        data_slice: slice, optional,
            Slice object to select a portion of the data. Default is slice(None).
        features: list or None, optional
            List of features to select. Default is None.
            If `None`, no features are selected, if 'all', all features are selected. Otherwise, the indices in the list are selected.
        seed: int, optional,
            Random seed for reproducibility. Default is 1.
        regression_feature: int, optional,
            Index of the regression target (the feature will be taken from data['jet_features'][...,regression_feature]). Default is -1.
        inputs: list, optional,
            List of input types to load. Default is ["coords", "features", "mask"].
        particles: int, optional,
            Number of particles per jet. Default is 30.
        njets: int, optional,
            Number of jets > 1. Default is 2.
        discriminator: Discriminator, optional,
            Discriminator object to apply to the data. Default is None.
        log: bool, optional,
            If True, the regression target is transformed to log10(target). Default is True.
        shift_mean: bool or float, optional,
            If True, the mean of the regression target is subtracted from the target. Default is True.
            If a float, this value is subtracted from the target.
        """
        super().__init__(workers=1, use_multiprocessing=False)
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
        if(shift_mean==True):
            self.shift = np.mean(self.target)
            self.target -= self.shift
        elif type(shift_mean) == float:
            self.shift = shift_mean
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
        return tuple(self.data[jet][s][start:end] for jet in self.jets for s in self.inputs), self.target[start:end]

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
    
class MergedRegressionDataLoader(RegressionDataLoader):
    """
    Similar to RegressionDataLoader, but merges the data from all jets into one array.
    """
    def __init__(self, data, batch_size:int, N_data, data_slice=None, features=None, seed:int=1, regression_feature:int=-1,
                 inputs:list=["coords", "features", "mask"], particles:int=30, njets:int=2,discriminator:Discriminator=None,
                 log:bool=True, shift_mean:bool=True):
        super().__init__(data, batch_size, N_data, data_slice, features, seed, regression_feature,
                 inputs, particles, njets,discriminator, log, shift_mean)
        self.data = {s: np.concatenate([self.data[jet][s] for jet in self.jets], axis=1) for s in self.inputs}

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        return tuple(self.data[s][start:end] for s in self.inputs), self.target[start:end]

