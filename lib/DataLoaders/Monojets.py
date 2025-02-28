from .Discriminator import Discriminator
import numpy as np
from .DataLoader import BaseDataLoader
from .DecoderFeatures import DecoderFeature
import h5py
from time import time
from typing import Dict
import gc

def _check_slices(slices):
    slices = slices if isinstance(slices, (tuple,list)) else [slices]
    if not all([s.step is None or s.step == 1 for s in slices]):
        raise ValueError("All slices should have step 1 or None")
    intervals = [(s.start, s.stop) for s in slices]
    intervals.sort()
    for i in range(len(intervals) - 1):
        if intervals[i][1] > intervals[i+1][0]:
            raise ValueError(f"Slices {intervals[i]} and {intervals[i+1]} overlap!")
    return slices

class IADDataLoader_SingleJet(BaseDataLoader):
    """
    A version of IADDataLoaderV2 that only supports a single jet and IAD version of `DataLoader_SingleJet`.
    Takes a simulated background, background and signal dataset and combines them into two other datasets:
    1. Pure background (size: N_simulated)
    2. Background + some signal contamination (size: N_background+N_signal)

    - Supports different simulated background and data background
    - Specifically for single-jet datasets
    - Supports discriminator
    - Supports option to first take a slice from the data before selecting the events according to `N_background` and `N_signal`
        Useful for splitting the data into training and validation sets.
    - Supports oversampling
    - Supports adding noise to the features
    - Supports including the true labels in the output (for validation/testing)
    """
    def __init__(self, sim_bg:Dict[np.ndarray], data_sn:Dict[np.ndarray], data_bg:Dict[np.ndarray], batch_size:int, N_simulated:int, N_background:int, N_signal:int, features:list=None, particles:int=30, seed:int=1,
                 discriminator:Discriminator=None, inputs:list=["coords", "features", "mask"],
                 sim_bg_slice=None, data_sn_slice=None, data_bg_slice=None, oversampling:str=None,
                 noise_features:int=None, noise_param:tuple=(0,1), include_true_labels:bool=False, do_shuffle:bool=True):
        """
        params
        ------
        sim_bg: dict,
            Simulated background data dictionary.
        data_sn: dict,
            Signal data dictionary.
        data_bg: dict,
            Background data dictionary.
        batch_size: int,
            Size of each batch.
        N_simulated: int,
            Number of simulated events to load.
        N_background: int,
            Number of background events to load.
        N_signal: int,
            Number of signal events to load.
        features: list or None, optional
            List of features to select. Default is None.
            If `None`, no features are selected, if 'all', all features are selected. Otherwise, the indices in the list are selected.
        seed: int, optional,
            Random seed for reproducibility. Default is 1.
        discriminator: Discriminator or None, optional,
            Discriminator object to filter events. Default is None.
        inputs: list, optional,
            List of input types to load. Default is ["coords", "features", "mask"].
        decoder_inputs: list of callables or None, optional,
            List of decoder input functions. Default is None.
        particles: int, optional,
            Number of particles per jet. Default is 30.
        sim_bg_slice: slice or list of slices, optional,
            Slice object to select a portion of the simulated background data. Default is slice(None).
        data_sn_slice: slice or list of slices, optional,
            Slice object to select a portion of the signal data. Default is slice(None).
        data_bg_slice: slice or list of slices, optional,
            Slice object to select a portion of the background data. Default is slice(None).
        oversampling: str, optional,
            Optional oversampling method, see `oversampling.py` for more information. Default is None.
        noise_features: int, optional,
            Number of noise features to add to `features`. Default is None.
        noise_param: tuple, optional,
            Parameters for the noise distribution. Default is (0,1).
        include_true_labels: bool, optional,
            Include true labels in the output. Default is False.
            Should be used for the validation set.
        do_shuffle: bool, optional,
            Shuffle the data. Default is True.
        """
        super().__init__(seed)
        self.inputs = inputs
        self.include_true_labels = include_true_labels

        sim_bg_slice = _check_slices(sim_bg_slice)
        data_sn_slice = _check_slices(data_sn_slice)
        data_bg_slice = _check_slices(data_bg_slice)

        #do some checks
        s = inputs[0]
        assert(data_bg[s].shape[0] >= N_background and data_sn[s].shape[0] >= N_signal and sim_bg[s].shape[0] >= N_simulated), \
            "Dataset is smaller than number of events to be read from it"
        assert all([data_bg[k].shape[1:] == data_sn[k].shape[1:] for k in inputs]), \
            "Shapes of background and signal don't match"

        # n_signals per batch contains a list of numpy arrays of size (100,) with each index's value corresponding to
        # the number of batches with a number of signal events equal to that index. ([0,2,1]) means 2 batches with 1 signal, 1 batches with 2 signals
        self.n_signal_per_batch = [] 

        #apply the discriminator
        if(discriminator is not None):
            passed_sim_bg_disc = discriminator.apply(sim_bg)
            idx_sim = np.where(np.concatenate([passed_sim_bg_disc[s] for s in sim_bg_slice], axis=0))[0]
            passed_data_bg_disc = discriminator.apply(data_bg)
            idx_bg = np.where(np.concatenate([passed_data_bg_disc[s] for s in data_bg_slice], axis=0))[0]
            passed_data_sn_disc = discriminator.apply(data_sn)
            idx_sn = np.where(np.concatenate([passed_data_sn_disc[s] for s in data_sn_slice], axis=0))[0]
        else:
            idx_sim = np.concatenate([
                np.arange(_slice.start or 0, _slice.stop or sim_bg[s].shape[0], _slice.step or 1)-(_slice.start or 0)
                for _slice in sim_bg_slice], axis=0)
            idx_bg = np.concatenate([
                np.arange(_slice.start or 0, _slice.stop or data_bg[s].shape[0], _slice.step or 1)-(_slice.start or 0)
                for _slice in data_bg_slice],axis=0)
            idx_sn = np.concatenate([
                np.arange(_slice.start or 0, _slice.stop or data_sn[s].shape[0], _slice.step or 1)-(_slice.start or 0)
                for _slice in data_sn_slice],axis=0)
            
        n_simulated_in_data = len(idx_sim)
        n_background_in_data = len(idx_bg)
        n_signal_in_data = len(idx_sn)
        print(f"{n_simulated_in_data} simulated, {n_background_in_data} background and {n_signal_in_data} signal events passed the discriminator and/or slice")

        #find background and signal events
        assert n_simulated_in_data>=N_simulated, f"Not enough simulated events ({n_simulated_in_data}<{N_simulated})"
        assert n_background_in_data>=N_background, f"Not enough background events ({n_background_in_data}<{N_background})"
        assert n_signal_in_data>=N_signal, f"Not enough signal events ({n_signal_in_data}<{N_signal})"
        if(N_simulated < 0): N_simulated = n_simulated_in_data
        if(N_background < 0): N_background = n_background_in_data
        if(N_signal < 0): N_signal = n_signal_in_data
        num_events = N_simulated+N_background+N_signal
        
        #initialize data
        if((features is None or len(features)==0) and "features" in self.inputs):
            self.inputs.remove("features")
        particles = particles or sim_bg[self.inputs[0]].shape[1]
        self.data = {s:np.empty((num_events, particles, *sim_bg[s].shape[2:]), dtype=np.float32) for s in self.inputs}
        
        def add_to_data(from_dataset, from_slice, to_slice, indices):
            for x in self.inputs:
                self.data[x][to_slice] = np.array(from_dataset[x][from_slice, :particles], dtype=np.float32)[indices]
        
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

        self.labels = np.zeros((num_events, 2),dtype=int)
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
            num_events = self.labels.shape[0]

        self.finish_init(batch_size, num_events//batch_size, num_events)
        if(do_shuffle):
            self.shuffle()
    
    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        self.n_signal_per_batch[-1][min(np.sum(self.true_labels[start:end,1]),self.max_signal_per_batch)] += 1
        if not self.include_true_labels:
            return tuple(self.data[key][start:end] for key in self.input_order), self.labels[start:end]
        return tuple(self.data[key][start:end] for key in self.input_order), (self.labels[start:end], self.true_labels[start:end])

#just for backward-compatibility
class IADDataLoader_MonoJets(IADDataLoader_SingleJet):
    pass

class DataLoader_SingleJet(BaseDataLoader):
    """
    A singlejet version of `JetDataLoaderV2`
    Takes a background and signal dataset and combines them into two other datasets:
    1. Pure background (size: N_background)
    2. Pure signal contamination (size: N_signal)

    - Specifically for single-jet datasets
    - Supports discriminator
    - Supports option to first take a slice from the data before selecting the events according to `N_background` and `N_signal`
        Useful for splitting the data into training and validation sets.
    - Supports oversampling
    - Supports adding noise to the features
    - Supports decoder inputs
    """
    def __init__(self, data_sn:Dict[np.ndarray], data_bg:Dict[np.ndarray], batch_size:int, N_background:int, N_signal:int,
                 features:list=None, particles:int=30, seed:int=1, discriminator:Discriminator=None,
                 inputs:list=["coords", "features", "mask"],
                 data_sn_slice=None, data_bg_slice=None, oversampling:str=None,
                 noise_features:int=None, noise_param:tuple=(0,1),  decoder_inputs:list[DecoderFeature]=None, do_shuffle:bool=True):
        """
        params
        ------
        data_sn: dict,
            Signal data dictionary.
        data_bg: dict,
            Background data dictionary.
        batch_size: int,
            Size of each batch.
        N_background: int,
            Number of background events to load.
        N_signal: int,
            Number of signal events to load.
        features: list or None, optional
            List of features to select. Default is None.
            If `None`, no features are selected, if 'all', all features are selected. Otherwise, the indices in the list are selected.
        particles: int, optional,
            Number of particles per jet. Default is 30.
        seed: int, optional,
            Random seed for reproducibility. Default is 1.
        discriminator: Discriminator or None, optional,
            Discriminator object to filter events. Default is None.
        inputs: list, optional,
            List of input types to load. Default is ["coords", "features", "mask"].
        data_sn_slice: slice or list of slices, optional,
            Slice object to select a portion of the signal data. Default is slice(None).
        data_bg_slice: slice or list of slices, optional,
            Slice object to select a portion of the background data. Default is slice(None).
        oversampling: str, optional,
            Optional oversampling method, see `oversampling.py` for more information. Default is None.
        noise_features: int, optional,
            Number of noise features to add to `features`. Default is None.
        noise_param: tuple, optional,
            Parameters for the noise distribution. Default is (0,1).
        decoder_inputs: list of callables or None, optional,
            List of decoder input functions. Default is None.
        do_shuffle: bool, optional,
            Shuffle the data. Default is True.
        """
        super().__init__(seed)
        self.inputs = inputs

        data_sn_slice = _check_slices(data_sn_slice)
        data_bg_slice = _check_slices(data_bg_slice)

        #do some checks
        s = inputs[0]
        assert(data_bg[s].shape[0] >= N_background and data_sn[s].shape[0] >= N_signal), \
            "Dataset is smaller than number of events to be read from it"
        assert all([data_bg[k].shape[1:] == data_sn[k].shape[1:] for k in inputs]), \
            "Shapes of background and signal don't match"

        # n_signals per batch contains a list of numpy arrays of size (100,) with each index's value corresponding to
        # the number of batches with a number of signal events equal to that index. ([0,2,1]) means 2 batches with 1 signal, 1 batches with 2 signals
        self.n_signal_per_batch = [] 

        #apply the discriminator
        if(discriminator is not None):
            passed_data_bg_disc = discriminator.apply(data_bg)
            idx_bg = np.where(np.concatenate([passed_data_bg_disc[s] for s in data_bg_slice], axis=0))[0]
            passed_data_sn_disc = discriminator.apply(data_sn)
            idx_sn = np.where(np.concatenate([passed_data_sn_disc[s] for s in data_sn_slice], axis=0))[0]
        else:
            idx_bg = np.concatenate([
                np.arange(_slice.start or 0, _slice.stop or data_bg[s].shape[0], _slice.step or 1)-(_slice.start or 0)
                for _slice in data_bg_slice],axis=0)
            idx_sn = np.concatenate([
                np.arange(_slice.start or 0, _slice.stop or data_sn[s].shape[0], _slice.step or 1)-(_slice.start or 0)
                for _slice in data_sn_slice],axis=0)
            
        n_background_in_data = len(idx_bg)
        n_signal_in_data = len(idx_sn)
        print(f"{n_background_in_data} background and {n_signal_in_data} signal events passed the discriminator and/or slice")

        #find background and signal events
        assert n_background_in_data>=N_background, f"Not enough background events ({n_background_in_data}<{N_background})"
        assert n_signal_in_data>=N_signal, f"Not enough signal events ({n_signal_in_data}<{N_signal})"
        if(N_background < 0): N_background = n_background_in_data
        if(N_signal < 0): N_signal = n_signal_in_data
        num_events = N_background+N_signal
        
        #initialize data
        if((features is None or len(features)==0) and "features" in self.inputs):
            self.inputs.remove("features")
        elif(features == 'all'):
            features = list(range(data_bg["features"].shape[-1]))
        particles = particles or data_bg[self.inputs[0]].shape[1]
        self.data = {s:np.empty((num_events, particles, data_bg[s].shape[2] if s!='features' else len(features)), dtype=np.float32) for s in self.inputs}
    
        if decoder_inputs is None:
            self.decoder_features = False
            def add_to_data(from_dataset, from_slice, to_slice, indices):
                for x in self.inputs:
                    if x == "features":
                        self.data[x][to_slice] = np.stack([np.array(from_dataset[x][from_slice, :particles,i])[[indices]] for i in features], axis=-1)
                    else:
                        self.data[x][to_slice] = np.array(from_dataset[x][from_slice, :particles], dtype=np.float32)[indices]
        else:
            self.decoder_features = True
            self.data["decoder"] = np.empty((num_events, sum([dec.dimension for dec in decoder_inputs])), dtype=np.float32)
            dec_datasets = {x for d in decoder_inputs for x in d.required_datasets}
            def add_to_data(from_dataset, from_slice, to_slice, indices):
                temp_data = {x:np.array(from_dataset[x][from_slice, :particles], dtype=np.float32)[indices] for x in self.inputs}
                for x in temp_data.keys():
                    self.data[x][to_slice] = temp_data[x]
                dec_data = {x:np.array(from_dataset[x][from_slice])[indices] for x in dec_datasets}
                self.data["decoder"][to_slice] = np.concatenate([dec(temp_data, dec_data) for dec in decoder_inputs],axis=-1)

        if len(data_bg_slice)==1:
            #select the required number of background events
            idx = np.sort(self.rnd.choice(idx_bg, size=N_background, replace=False))
            add_to_data(data_bg, data_bg_slice[0], slice(None, N_background), idx)
        else:
            raise ValueError("Only one slice for background is supported")
        if len(data_sn_slice)==1:
            #select the required number of signal events
            idx = np.sort(self.rnd.choice(idx_sn, size=N_signal, replace=False))
            add_to_data(data_sn, data_sn_slice[0], slice(N_background, None), idx)
        else:
            raise ValueError("Only one slice for background is supported")

        #if(features != 'all' and features is not None):
            #select the required features
        #   self.data["features"] = self.data["features"][:,:particles,features] #yes, not very efficient. I know
        if(noise_features):
            rng2 = np.random.default_rng(seed)
            self.data["features"] = np.concatenate((self.data["features"], rng2.normal(*noise_param,size=(*self.data["features"].shape[:-1],noise_features))), axis=-1)

        self.labels = np.zeros((num_events, 2),dtype=int)
        self.labels[:N_background,0] = 1 #background labels
        self.labels[N_background:,1] = 1 #signal labels
               
        if(oversampling):
            from .oversample import oversample
            if(N_background > N_signal):
                #oversample the data
                idx = oversample(N_signal, N_background, mode=oversampling, shuffle=False, rng=self.rnd)
                for key in self.data.keys():
                    d = self.data[key][N_background:]
                    d = d[idx]
                    self.data[key] = np.concatenate((self.data[key][:N_background], d),axis=0)
                self.labels = np.concatenate((self.labels[:N_background], self.labels[N_background:][idx]),axis=0)
            elif(N_background < N_signal):
                #oversample simulated
                idx = oversample(N_background, N_signal, mode=oversampling, shuffle=False, rng=self.rnd)
                for key in self.data.keys():
                    d = self.data[key][:N_background]
                    d = d[idx]
                    self.data[key] = np.concatenate((d, self.data[key][N_background:]),axis=0)
                self.labels = np.concatenate((self.labels[:N_background][idx], self.labels[N_background:]),axis=0)
            num_events = self.labels.shape[0]

        self.finish_init(batch_size, num_events//batch_size, num_events)
        if(do_shuffle):
            self.shuffle()
    
    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        #self.n_signal_per_batch[-1][min(np.sum(self.labels[start:end,1]),self.max_signal_per_batch)] += 1
        ret = tuple(self.data[key][start:end] for key in self.inputs)
        if self.decoder_features: ret  = ret + (self.data["decoder"][start:end],)
        return ret, self.labels[start:end]

class MultiDataLoader_SingleJet(BaseDataLoader):
    """
    Loads multiple datasets and combines them into two other datasets:
    1. Pure background (size: N_background)
    2. Pure signal contamination (size: N_signal)
    The provided datasets are rotated through in a round-robin fashion after each epoch.

    - Specifically for multiple single-jet datasets
    - Supports option to first take a slice from the data before selecting the events according to `N_background` and `N_signal`
        Useful for splitting the data into training and validation sets.
    - Support adding noise to the features
    - Supports decoder inputs
    - Does not support discriminator
    - Does not support oversampling
    """
    def __init__(self, n_datasets:int, sn_file_pattern:str, bg_file_pattern:str, batch_size:int, N_background:int, N_signal:int,
                 data_sn_slice:slice=None, data_bg_slice:slice=None,
                 features:list=None, particles:int=30, seed:int=1, inputs:list=["coords", "features", "mask"],
                 noise_features:int=None, noise_param:tuple=(0,1),  decoder_inputs:list[DecoderFeature]=None):
        """
        params
        ------
        n_datasets: int,
            Number of datasets to load.
        sn_file_pattern: str,
            Signal file pattern. '?' will be replaced by the dataset number.
            Note that `N_signal` and `data_sn_slice` are applied to each file.
        bg_file_pattern: str,
            Background file pattern. '?' will be replaced by the dataset number.
            Note that `N_background` and `data_bg_slice` are applied to each file.
        batch_size: int,
            Size of each batch.
        N_background: int,
            Number of background events to load (for each file).
        N_signal: int,
            Number of signal events to load (for each file).
        data_sn_slice: slice or list of slices, optional,
            Slice object to select a portion of the signal data. Default is slice(None).
        data_bg_slice: slice or list of slices, optional,
            Slice object to select a portion of the background data. Default is slice(None).
        features: list or None, optional
            List of features to select. Default is None.
            If `None`, no features are selected, if 'all', all features are selected. Otherwise, the indices in the list are selected.
        particles: int, optional,
            Number of particles per jet. Default is 30.
        seed: int, optional,
            Random seed for reproducibility. Default is 1.
        inputs: list, optional,
            List of input types to load. Default is ["coords", "features", "mask"].
        noise_features: int, optional,
            Number of noise features to add to `features`. Default is None.
        noise_param: tuple, optional,
            Parameters for the noise distribution. Default is (0,1).
        decoder_inputs: list of callables or None, optional,
            List of decoder input functions. Default is None.
        """
        super().__init__(seed, workers=2, use_multiprocessing=True)
        self.inputs = inputs
        self.sn_file_pattern = sn_file_pattern
        self.bg_file_pattern = bg_file_pattern
        
        assert batch_size%2 == 0, "Batch size must be even"
        self.batch_size = batch_size
        self.N_background = N_background
        self.N_signal = N_signal

        if((features is None or len(features)==0) and "features" in self.inputs):
            self.inputs.remove("features")
        self.features = features
        self.particles = particles
        self.seed = seed
        self.inputs = inputs
        self.data_sn_slice = data_sn_slice or slice(None)
        self.data_bg_slice = data_bg_slice or slice(None)
        assert self.data_bg_slice.step == 1 or self.data_bg_slice.step is None, "Background slice step must be 1 (or None)"
        assert self.data_sn_slice.step == 1 or self.data_sn_slice.step is None, "Signal slice step must be 1 (or None)"
        self.noise_features = noise_features
        self.noise_param = noise_param
        self.decoder_inputs = decoder_inputs
        self.current_epoch = 0
        self.loaded_chunk = None
        self.n_datasets = n_datasets
        self.data_bg = None
        self.data_sn = None
        self.time = 0
        self.cycle_dataset()

    def cycle_dataset(self):
        tic = time()
        print("REFRESHING DATA")
        if hasattr(self, 'data'):
            del self.data
            gc.collect()
        self.current_epoch += 1
        self.seed += 1
        #do some checks
        s = self.inputs[0]
        if(self.data_bg is not None): self.data_bg.close()
        if(self.data_sn is not None): self.data_sn.close()
        self.data_bg = h5py.File(self.bg_file_pattern.replace('?', str(self.current_epoch%self.n_datasets)), 'r')
        self.data_sn = h5py.File(self.sn_file_pattern.replace('?', str(self.current_epoch%self.n_datasets)), 'r')
        assert(self.data_bg[s].shape[0] >= self.N_background and self.data_sn[s].shape[0] >= self.N_signal), \
            "Dataset is smaller than number of events to be read from it"
        assert all([self.data_bg[k].shape[1:] == self.data_sn[k].shape[1:] for k in self.inputs]), \
            "Shapes of background and signal don't match"
        if(self.features == 'all'):
            self.features = list(range(self.data_bg["features"].shape[-1]))
        
        self.n_signal_per_batch = [] 

        n_background_in_data = (self.data_bg_slice.stop or self.data_bg["signal"].shape[0])-(self.data_bg_slice.start or 0)
        n_signal_in_data = (self.data_sn_slice.stop or self.data_sn["signal"].shape[0])-(self.data_sn_slice.start or 0)
        print(f"{n_background_in_data} background and {n_signal_in_data} signal events passed the slice")

        #find background and signal events
        assert n_background_in_data>=self.N_background, f"Not enough background events ({n_background_in_data}<{self.N_background})"
        assert n_signal_in_data>=self.N_signal, f"Not enough signal events ({n_signal_in_data}<{self.N_signal})"
        if(self.N_background < 0): self.N_background = n_background_in_data
        if(self.N_signal < 0): self.N_signal = n_signal_in_data
        self.N = self.num_events = self.N_background+self.N_signal

        self.particles = self.particles or self.data_bg[self.inputs[0]].shape[1]
        self.N_batches = self.num_events//self.batch_size
        self.bg_per_batch = int(self.N_background/self.N_batches)
        self.sn_per_batch = int(self.N_signal/self.N_batches)
        new_batch_size = self.bg_per_batch+self.sn_per_batch
        print(f"Using {self.bg_per_batch} background and {self.sn_per_batch} signal events per batch")
        if self.batch_size != new_batch_size:
            print(f"Batch size changed from {self.batch_size} to {new_batch_size}")
        self.batch_size = new_batch_size

        if self.decoder_inputs is None:
            self.decoder_features = False
            def add_to_data(from_dataset, from_slice, to_slice):
                for x in self.inputs:
                    if x == "features":
                        self.data[x][to_slice] = np.stack([np.array(from_dataset[x][from_slice, :self.particles,i]) for i in self.features], axis=-1)
                    else:
                        self.data[x][to_slice] = np.array(from_dataset[x][from_slice, :self.particles], dtype=np.float32)
        else:
            self.decoder_features = True
            dec_datasets = {x for d in self.decoder_inputs for x in d.required_datasets}
            def add_to_data(from_dataset, from_slice, to_slice):
                temp_data = {x:np.array(from_dataset[x][from_slice, :self.particles], dtype=np.float32) for x in self.inputs}
                for x in temp_data.keys():
                    self.data[x][to_slice] = temp_data[x]
                dec_data = {x:np.array(from_dataset[x][from_slice]) for x in dec_datasets}
                self.data["decoder"][to_slice] = np.concatenate([dec(temp_data, dec_data) for dec in self.decoder_inputs],axis=-1)
        
        self.add_to_data = add_to_data
        self.time += time()-tic

    def load_batch(self, idx):
        #first see which batches should be loaded
        #initialize data
        tic = time()
        n_events = self.bg_per_batch+self.sn_per_batch
        self.data = {s:np.empty((n_events, self.particles, self.data_bg[s].shape[2] if s!='features' else len(self.features)), dtype=np.float32) for s in self.inputs}
        if self.decoder_features is None:
            self.data["decoder"] = np.empty((n_events, sum([dec.dimension for dec in self.decoder_inputs])), dtype=np.float32)
            
        #find the indices of background and signal events
        #note that these indices are sorted!
        #select the required number of background events
        bg_slice = slice(idx*self.bg_per_batch, (idx+1)*self.bg_per_batch)
        self.add_to_data(self.data_bg, bg_slice, slice(None, self.bg_per_batch))
        #select the required number of signal events
        sn_slice = slice(idx*self.sn_per_batch, (1+idx)*self.sn_per_batch)
        self.add_to_data(self.data_sn, sn_slice, slice(self.bg_per_batch, None))
        
        if(self.noise_features):
            rng2 = np.random.default_rng(self.seed)
            self.data["features"] = np.concatenate((self.data["features"], rng2.normal(*self.noise_param,size=(*self.data["features"].shape[:-1],self.noise_features))), axis=-1)

        self.labels = np.zeros((n_events, 2),dtype=int)
        self.labels[:self.bg_per_batch,0] = 1 #background labels
        self.labels[self.bg_per_batch:,1] = 1 #signal labels
        self.time += time()-tic
    
    def shuffle(self):
        pass#return super().shuffle()

    def on_epoch_end(self):
        self.cycle_dataset()
        #self.load_chunk(0)
        super().on_epoch_end()

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        #first calculate in which chunk this batch is
        self.load_batch(idx)
        ret = tuple(self.data[key] for key in self.inputs)
        if self.decoder_features: ret  = ret + (self.data["decoder"],)
        return ret, self.labels
    
    def stop(self):
        super().stop()
        self.data_bg.close()
        self.data_sn.close()
        gc.collect()