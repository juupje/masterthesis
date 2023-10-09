from .DataLoader import DataLoader
from .Discriminator import Discriminator
import numpy as np
from time import time
from utils.misc import slice_to_indices
import utils.shuffling as shuffling
from .JetDataLoader import IADDataLoaderV2
'''
class DataLoader_MonoJets(tf.keras.utils.Sequence):
    def __init__(self, data_sn, data_bg, batch_size:int, N_background:int, N_signal:int, features:list=None, particles:int=30, seed:int=1,
                 inputs:dict={"4mom": "Particles", "features": "Data", "mask": None}, discriminator:Discriminator=None, coords:str="coords",
                 data_sn_slice=None, data_bg_slice=None, oversampling:str=None, noise_features:int=None, noise_param:tuple=(0,1), include_true_labels:bool=False):
        """
        Takes a background and signal dataset and combines them into two other datasets:
        1. Pure background (size: N_background)
        2. Background + some signal contamination (size: N_background+N_signal)
        Note, that the background dataset should contain at least 2*N_Background events.
        """
        self.input_order = ["coords", "features", "mask"]
        self.stopped = False
        self.rnd = np.random.default_rng(seed)
        self.include_true_labels = include_true_labels

        tic = time()
        #do some checks
        s = inputs[list(inputs.keys())[0]]
        assert(data_bg[s].shape[0] >= N_background and data_sn[s].shape[0] >= N_signal and sim_bg[s].shape[0] >= N_simulated), \
            "Dataset is smaller than number of events to be read from it"
        assert all([data_bg[k].shape[1:] == data_sn[k].shape[1:] for k in inputs.values() if k is not None]), \
            "Shapes of background and signal don't match"

        # n_signals per batch contains a list of numpy arrays of size (100,) with each index's value corresponding to
        # the number of batches with a number of signal events equal to that index. ([0,2,1]) means 2 batches with 1 signal, 1 batches with 2 signals
        self.n_signal_per_batch = [] 

        #apply the discriminator
        if(discriminator is not None):
            idx_sim = np.where(discriminator.apply(sim_bg)[sim_bg_slice])[0]
            idx_bg = np.where(discriminator.apply(data_bg)[data_bg_slice])[0]
            idx_sn = np.where(discriminator.apply(data_sn)[data_sn_slice])[0]
        else:
            idx_sim = np.arange(sim_bg_slice.start or 0, sim_bg_slice.stop or sim_bg[s].shape[0], sim_bg_slice.step or 1)-(sim_bg_slice.start or 0)
            idx_bg = np.arange(data_bg_slice.start or 0, data_bg_slice.stop or data_bg[s].shape[0], data_bg_slice.step or 1)-(data_bg_slice.start or 0)
            idx_sn = np.arange(data_sn_slice.start or 0, data_sn_slice.stop or data_sn[s].shape[0], data_sn_slice.step or 1)-(data_sn_slice.start or 0)
            
        n_simulated_in_data = len(idx_sim)
        n_background_in_data = len(idx_bg)
        n_signal_in_data = len(idx_sn)
        print(f"{n_simulated_in_data} simulated, {n_background_in_data} background and {n_signal_in_data} signal events passed the discriminator and/or slice")

        #find background and signal events
        assert n_simulated_in_data>=N_simulated, f"Not enough background events ({n_simulated_in_data}<{N_simulated})"
        assert n_background_in_data>=N_background, f"Not enough background events ({n_background_in_data}<{N_background})"
        assert n_signal_in_data>=N_signal, f"Not enough signal events ({n_signal_in_data}<{N_signal})"
        if(N_simulated < 0): N_simulated = n_simulated_in_data
        if(N_background < 0): N_background = n_background_in_data
        if(N_signal < 0): N_signal = n_signal_in_data
        self.N = N_simulated+N_background+N_signal
        
        #initialize data
        particles = particles or sim_bg[inputs["features"]].shape[1]
        self.data = {"coords":np.empty((self.N, particles, 4 if use_4mom else 2), dtype=np.float32),
                     "mask": np.empty((self.N, particles, 1), dtype=np.float32)}
        if(features):
            self.data["features"] = np.empty((self.N, particles, sim_bg[inputs["features"]].shape[2]), dtype=np.float32)
        else:
            self.input_order.remove("features")
        def add_to_data(from_dataset, from_slice, to_slice, indices):
            #first, add the features
            if(features):
                self.data["features"][to_slice] = np.array(from_dataset[inputs['features']][from_slice,:particles], dtype=np.float32)[indices]
            
            if(use_4mom):
                #add the 4 momentum
                self.data["coords"][to_slice] = np.array(from_dataset[inputs['4mom']][from_slice, :particles], dtype=np.float32)[indices]
            else:
                #add the (eta,phi) coordinates
                if("coords" in inputs and inputs["coords"]): #are the coords stored in a special dataset?
                    self.data["coords"][to_slice] = np.array(from_dataset[inputs['coords']][from_slice, :particles], dtype=np.float32)[indices]
                else: #nope, we need to extract them from the features
                    self.data["coords"][to_slice] = np.array(from_dataset[inputs['features']][from_slice,:particles,:2], dtype=np.float32)[indices]
            #add the mask
            if("mask" in inputs and inputs["mask"]):
                self.data["mask"][to_slice] = np.expand_dims(np.array(from_dataset[inputs["mask"]][from_slice,:particles], dtype=np.float32)[indices],axis=2)
            else:
                if(use_4mom):
                    self.data["mask"][to_slice] = np.expand_dims(self.data["coords"][to_slice,:,0]==0, axis=2).astype(np.int8)
                else:
                    self.data["mask"][to_slice] = np.expand_dims(np.logical_and(self.data["features"][to_slice,:particles,0]==0, self.data["features"][to_slice,:,1]==0).astype(np.int8), axis=2)
        
        #select the required number of simulated events
        idx = np.sort(self.rnd.choice(idx_sim, size=N_simulated, replace=False))
        add_to_data(sim_bg, sim_bg_slice, slice(None, N_simulated, None), idx)
        
        #select the required number of background events
        idx = np.sort(self.rnd.choice(idx_bg, size=N_background, replace=False))
        add_to_data(data_bg, data_bg_slice, slice(N_simulated, N_simulated+N_background, None), idx)

        #select the required number of signal events
        idx = np.sort(self.rnd.choice(idx_sn, size=N_signal, replace=False))
        add_to_data(data_sn, data_sn_slice, slice(N_simulated+N_background, None, None), idx)

        if(features != 'all' and features is not None):
            #select the required features
           self.data["features"] = self.data["features"][:,:particles,features] #yes, not very efficient. I know
        if(noise_features):
            rng2 = np.random.default_rng(seed)
            self.data["features"] = np.concatenate((self.data["features"], rng2.normal(*noise_param,size=(*self.data["features"].shape[:-1],1))), axis=-1)

        self.labels = np.zeros((self.N, 2),dtype=int)
        self.true_labels = np.zeros_like(self.labels)
        self.labels[:N_simulated,0] = 1 #background labels
        self.labels[N_simulated:,1] = 1 #contaminated labels
        self.true_labels[:N_simulated+N_background,0] = 1 # true background
        self.true_labels[N_simulated+N_background:,1] = 1 # true signal
        
        N_data = N_background+N_signal
        if(oversampling):
            from .oversample import oversample
            if(N_simulated > N_data):
                #oversample the data
                idx = oversample(N_data, N_simulated, mode=oversampling, shuffle=False, rng=self.rnd)
                for key in self.data.keys():
                    d = self.data[key][N_simulated:]
                    d = d[idx]
                    self.data[key] = np.concatenate((self.data[key][:N_simulated], d),axis=0)
                self.labels = np.concatenate((self.labels[:N_simulated], self.labels[N_simulated:][idx]),axis=0)
                self.true_labels = np.concatenate((self.true_labels[:N_simulated], self.true_labels[N_simulated:][idx]),axis=0)
            elif(N_simulated < N_data):
                #oversample simulated
                idx = oversample(N_simulated, N_data, mode=oversampling, shuffle=False, rng=self.rnd)
                for key in self.data.keys():
                    d = self.data[key][:N_simulated]
                    d = d[idx]
                    self.data[key] = np.concatenate((d, self.data[key][N_simulated:]),axis=0)
                self.labels = np.concatenate((self.labels[:N_simulated][idx], self.labels[N_simulated:]),axis=0)
                self.true_labels = np.concatenate((self.true_labels[:N_simulated][idx], self.true_labels[N_simulated:]),axis=0)
            self.N = self.labels.shape[0]

        self.N_batches = (self.N)//batch_size
        self.batch_size = batch_size
        self.SR = N_signal/N_data
        self.time = time()-tic
        self.shuffle_time = 0
        self.shuffle()
    
    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        self.n_signal_per_batch[-1][min(np.sum(self.true_labels[start:end,1]),self.max_signal_per_batch)] += 1
        if not self.include_true_labels:
            return [[self.data[key][start:end] for key in self.input_order], self.labels[start:end]]
        return [[self.data[key][start:end] for key in self.input_order], [self.labels[start:end], self.true_labels[start:end]]]
'''
class IADDataLoader_MonoJets(IADDataLoaderV2):
    def __init__(self, sim_bg, data_sn, data_bg, batch_size:int, N_simulated:int, N_background:int, N_signal:int, features:list=None, particles:int=30, seed:int=1, discriminator:Discriminator=None, use_4mom:bool=True,
                 sim_bg_slice=None, data_sn_slice=None, data_bg_slice=None, oversampling:str=None, noise_features:int=None, noise_param:tuple=(0,1), include_true_labels:bool=False, do_shuffle:bool=True):
        """
        Takes a background and signal dataset and combines them into two other datasets:
        1. Pure background (size: N_background)
        2. Background + some signal contamination (size: N_background+N_signal)
        Note, that the background dataset should contain at least 2*N_Background events.
        """
        self.input_order = ["coordinates", "features", "mask"]
        self.stopped = False
        self.rnd = np.random.default_rng(seed)
        self.include_true_labels = include_true_labels

        tic = time()
        #do some checks
        s = "Particles"
        assert(data_bg[s].shape[0] >= N_background and data_sn[s].shape[0] >= N_signal and sim_bg[s].shape[0] >= N_simulated), \
            "Dataset is smaller than number of events to be read from it"
        assert all([data_bg[k].shape[1:] == data_sn[k].shape[1:] for k in ["Data","Particles"]]), \
            "Shapes of background and signal don't match"

        # n_signals per batch contains a list of numpy arrays of size (100,) with each index's value corresponding to
        # the number of batches with a number of signal events equal to that index. ([0,2,1]) means 2 batches with 1 signal, 1 batches with 2 signals
        self.n_signal_per_batch = [] 

        #apply the discriminator
        if(discriminator is not None):
            idx_sim = np.where(discriminator.apply(sim_bg)[sim_bg_slice])[0]
            idx_bg = np.where(discriminator.apply(data_bg)[data_bg_slice])[0]
            idx_sn = np.where(discriminator.apply(data_sn)[data_sn_slice])[0]
        else:
            idx_sim = np.arange(sim_bg_slice.start or 0, sim_bg_slice.stop or sim_bg[s].shape[0], sim_bg_slice.step or 1)-(sim_bg_slice.start or 0)
            idx_bg = np.arange(data_bg_slice.start or 0, data_bg_slice.stop or data_bg[s].shape[0], data_bg_slice.step or 1)-(data_bg_slice.start or 0)
            idx_sn = np.arange(data_sn_slice.start or 0, data_sn_slice.stop or data_sn[s].shape[0], data_sn_slice.step or 1)-(data_sn_slice.start or 0)
            
        n_simulated_in_data = len(idx_sim)
        n_background_in_data = len(idx_bg)
        n_signal_in_data = len(idx_sn)
        print(f"{n_simulated_in_data} simulated, {n_background_in_data} background and {n_signal_in_data} signal events passed the discriminator and/or slice")

        #find background and signal events
        assert n_simulated_in_data>=N_simulated, f"Not enough background events ({n_simulated_in_data}<{N_simulated})"
        assert n_background_in_data>=N_background, f"Not enough background events ({n_background_in_data}<{N_background})"
        assert n_signal_in_data>=N_signal, f"Not enough signal events ({n_signal_in_data}<{N_signal})"
        if(N_simulated < 0): N_simulated = n_simulated_in_data
        if(N_background < 0): N_background = n_background_in_data
        if(N_signal < 0): N_signal = n_signal_in_data
        self.N = N_simulated+N_background+N_signal
        
        #initialize data
        particles = particles or sim_bg["Data"].shape[1]
        self.data = {"coordinates":np.empty((self.N, particles, 4 if use_4mom else 2), dtype=np.float32),
                     "mask": np.empty((self.N, particles, 1), dtype=np.float32)}
        if(features):
            self.data["features"] = np.empty((self.N, particles, sim_bg["Data"].shape[2]), dtype=np.float32)
        else:
            self.input_order.remove("features")
        
        def add_to_data(from_dataset, from_slice, to_slice, indices):
            #Add the coordinates
            if(use_4mom):
                #add the 4 momentum
                self.data["coordinates"][to_slice] = np.array(from_dataset["Particles"][from_slice, :particles], dtype=np.float32)[indices]
            else:
                self.data["coordinates"][to_slice] = np.array(from_dataset["Data"][from_slice,:particles,:2], dtype=np.float32)[indices]
            #first, add the features
            if(features):
                self.data["features"][to_slice] = np.array(from_dataset["Data"][from_slice,:particles], dtype=np.float32)[indices]
            #add the mask
            self.data["mask"][to_slice] = np.expand_dims(np.array(from_dataset["Particles"][from_slice,:particles,0], dtype=np.float32)[indices],axis=2)
        
        #select the required number of simulated events
        idx = np.sort(self.rnd.choice(idx_sim, size=N_simulated, replace=False))
        add_to_data(sim_bg, sim_bg_slice, slice(None, N_simulated, None), idx)
        
        #select the required number of background events
        idx = np.sort(self.rnd.choice(idx_bg, size=N_background, replace=False))
        add_to_data(data_bg, data_bg_slice, slice(N_simulated, N_simulated+N_background, None), idx)

        #select the required number of signal events
        idx = np.sort(self.rnd.choice(idx_sn, size=N_signal, replace=False))
        add_to_data(data_sn, data_sn_slice, slice(N_simulated+N_background, None, None), idx)

        if(features != 'all' and features is not None):
            #select the required features
           self.data["features"] = self.data["features"][:,:particles,features] #yes, not very efficient. I know
        if(noise_features):
            rng2 = np.random.default_rng(seed)
            self.data["features"] = np.concatenate((self.data["features"], rng2.normal(*noise_param,size=(*self.data["features"].shape[:-1],noise_features))), axis=-1)

        self.labels = np.zeros((self.N, 2),dtype=int)
        self.true_labels = np.zeros_like(self.labels)
        self.labels[:N_simulated,0] = 1 #background labels
        self.labels[N_simulated:,1] = 1 #contaminated labels
        self.true_labels[:N_simulated+N_background,0] = 1 # true background
        self.true_labels[N_simulated+N_background:,1] = 1 # true signal
        
        N_data = N_background+N_signal
        if(oversampling):
            from .oversample import oversample
            if(N_simulated > N_data):
                #oversample the data
                idx = oversample(N_data, N_simulated, mode=oversampling, shuffle=False, rng=self.rnd)
                for key in self.data.keys():
                    d = self.data[key][N_simulated:]
                    d = d[idx]
                    self.data[key] = np.concatenate((self.data[key][:N_simulated], d),axis=0)
                self.labels = np.concatenate((self.labels[:N_simulated], self.labels[N_simulated:][idx]),axis=0)
                self.true_labels = np.concatenate((self.true_labels[:N_simulated], self.true_labels[N_simulated:][idx]),axis=0)
            elif(N_simulated < N_data):
                #oversample simulated
                idx = oversample(N_simulated, N_data, mode=oversampling, shuffle=False, rng=self.rnd)
                for key in self.data.keys():
                    d = self.data[key][:N_simulated]
                    d = d[idx]
                    self.data[key] = np.concatenate((d, self.data[key][N_simulated:]),axis=0)
                self.labels = np.concatenate((self.labels[:N_simulated][idx], self.labels[N_simulated:]),axis=0)
                self.true_labels = np.concatenate((self.true_labels[:N_simulated][idx], self.true_labels[N_simulated:]),axis=0)
            self.N = self.labels.shape[0]

        self.N_batches = (self.N)//batch_size
        self.batch_size = batch_size
        self.SR = N_signal/N_data
        self.time = time()-tic
        self.shuffle_time = 0
        if(do_shuffle):
            self.shuffle()
    
    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        self.n_signal_per_batch[-1][min(np.sum(self.true_labels[start:end,1]),self.max_signal_per_batch)] += 1
        if not self.include_true_labels:
            return [[self.data[key][start:end] for key in self.input_order], self.labels[start:end]]
        return [[self.data[key][start:end] for key in self.input_order], [self.labels[start:end], self.true_labels[start:end]]]
