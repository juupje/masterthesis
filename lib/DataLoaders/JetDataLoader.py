from .DataLoader import BaseDataLoader
from .DecoderFeatures import DecoderFeature
from .Discriminator import Discriminator
import numpy as np
import logging
from typing import Dict

class JetDataLoader(BaseDataLoader):
    """
    Takes `N_background` background events and `N_signal` signal events from the given data.
    - Supports multiple jets
    - Supports decoder inputs
    - Supports discriminator
    - Supports option to first take a slice from the data before selecting the events according to `N_background` and `N_signal`
        Useful for splitting the data into training and validation sets.
    - The data is shuffled.
    - Supports replacing a jet with noise
    """
    def __init__(self, bg, sn, batch_size:int, N_background:int, N_signal:int, particles:int=30, features=None, seed:int=1,
                 bg_slice:slice=slice(None), sn_slice:slice=slice(None),
                 inputs:list=["coords", "features", "mask"], decoder_inputs:list[DecoderFeature]=None, njets:int=2, discriminator:Discriminator=None, **kwargs):
        """
        Initialize the JetDataLoader.

        Parameters
        -----------
        bg: dict,
            Background data dictionary.
        sn: dict,
            Signal data dictionary.
        batch_size: int,
            Size of each batch.
        N_background: int, 
            Number of background events to load.
        N_signal: int,
            Number of signal events to load.
        particles: int, optional,
            Number of particles per jet. Default is 30.
        features: list or None, optional
            List of features to select. Default is None.
        seed: int, optional,
            Random seed for reproducibility. Default is 1.
        bg_slice: slice, optional,
            Slice object to select a portion of the background data. Default is slice(None).
        sn_slice:  slice, optional,
            Slice object to select a portion of the signal data. Default is slice(None).
        inputs: list, optional,
            List of input types to load. Default is ["coords", "features", "mask"].
        decoder_inputs: list of callables or None, optional,
            List of decoder input functions. Default is None.
        njets: int, optional,
            Number of jets. Default is 2.
        discriminator: Discriminator or None, optional,
            Discriminator object to filter events. Default is None.
        **kwargs: Additional keyword arguments.
            'noise_jet': dict, optional,
                Add noise to a jet. Should contain the key 'index' with the jet index and 'type' with the noise type. Default is None.
                The type can be `mean` or `zero`:
                `mean` replaces a jet with its mean event (across all events, regardless of signal/background),
                `zero` replaces a jet with zeros.
        """        
        super().__init__(seed)
        print("kwargs:", kwargs)
        self.inputs = inputs.copy()
        self.njets = njets
        self.jets = [f"jet{idx}" for idx in range(1,1+njets)]
        N_data_bg = bg["signal"][bg_slice].shape[0]
        N_data_sn = sn["signal"][sn_slice].shape[0]
        self.logger = logging.getLogger(self.__class__.__name__)
        if(features is None):
            self.inputs.remove("features")

        #do some checks
        assert(N_data_bg >= N_background and N_data_sn >= N_signal), \
            "Dataset is smaller than number of events to be read from it"
        assert all([bg[jet][s].shape[2:] == sn[jet][s].shape[2:] for jet in self.jets for s in inputs]), \
            "Shapes of background and signal don't match"
        assert all([bg[jet][s].shape[1] >= particles for jet in self.jets for s in inputs]), \
            "Dataset does not contain enough particles per jet (background)"
        assert all([sn[jet][s].shape[1] >= particles for jet in self.jets for s in inputs]), \
            "Dataset does not contain enough particles per jet (signal)"
        #apply the discriminator
        disc = discriminator if discriminator is not None else Discriminator("jet_features/-1", None, None)
        idx_bg = np.where(disc.apply(bg, _slice=bg_slice))[0]
        idx_sn = np.where(disc.apply(sn, _slice=sn_slice))[0]
        n_background_in_data = len(idx_bg)
        n_signal_in_data = len(idx_sn)
        if(discriminator):
            print(f"{n_background_in_data} background and {n_signal_in_data} signal events passed the discriminator")

        #find background and signal events
        assert n_background_in_data>=N_background, "Not enough background events"
        assert n_signal_in_data>=N_signal, "Not enough signal events"
        if(N_background < 0): N_background = n_background_in_data
        if(N_signal < 0): N_signal = n_signal_in_data
        num_events = N_background+N_signal

        s_idx = np.zeros(num_events, dtype=bool) #default: all background
        s_idx[self.rnd.choice(num_events, N_signal, replace=False)] = True #these will be signals
        #initialize data
        self.data = {jet: {s:np.empty((num_events, particles, *bg[jet][s].shape[2:]), dtype=np.float32) for s in inputs} for jet in self.jets}
        
        #select the required number of background events
        idx_bg = np.sort(self.rnd.choice(idx_bg, size=N_background, replace=False))
        for s in inputs:
            for jet in self.jets:
                d = np.array(bg[f"{jet}/{s}"][bg_slice, :particles])[idx_bg]
                assert not(np.any(np.isnan(d)) or np.any(np.isinf(d))), "There is a nan value in the data"
                self.data[jet][s][~s_idx] = d #put the chosen events into the background spots

        #select the required number of signal events
        idx_sn = np.sort(self.rnd.choice(idx_sn, size=N_signal, replace=False))
        for s in inputs:
            for jet in self.jets:
                self.data[jet][s][s_idx] = np.array(sn[f"{jet}/{s}"][sn_slice,:particles])[idx_sn] #put the chosen events into the signal spots

        if decoder_inputs is not None:
            #first obtain the required data sets
            datasets = {x for d in decoder_inputs for x in d.required_datasets}
            for dataset in datasets:
                d = np.empty((num_events, *bg[dataset].shape[1:]), dtype=np.float32)
                d[~s_idx] = np.array(bg[dataset][bg_slice])[idx_bg]
                d[s_idx] = np.array(sn[dataset][sn_slice])[idx_sn]
                self.data[dataset] = d
            if(len(decoder_inputs)>1):
                self.data["decoder"] = np.concatenate([dec(self.data) for dec in decoder_inputs],axis=-1)
            else:
                self.data["decoder"] = decoder_inputs[0](self.data)
            for dataset in datasets:
                del self.data[dataset]
            assert self.data["decoder"].shape[0] == num_events and self.data["decoder"].ndim == 2, \
                f"Decoder features do not have the right shape. Expected ({num_events}, n), got {self.data['decoder'].shape}"
            self.decoder_inputs = True
        else:
            self.decoder_inputs = False

        if(features != 'all' and features is not None):
            #select the required features
            for jet in self.jets: self.data[jet]["features"] = self.data[jet]["features"][:,:,features] #yes, terrible code. I know
        
        if (noise_jet := kwargs.get("noise_jet",None)) is not None:
            #replace the jet with noise
            jet = noise_jet["index"]+1
            assert 0 < jet <= njets, f"Jet index {jet} is out of range"
            noise_type = noise_jet.get("type", 'mean')
            if(noise_type == 'mean'):
                self.logger.info(f"Replacing jet{jet} with the mean of all jet{jet}'s")
                for s in (s1 for s1 in inputs if s1!='mask'):
                    x = self.data[f"jet{jet}"][s]
                    self.data[f"jet{jet}"][s] = np.tile(np.mean(x,axis=0, keepdims=True), (x.shape[0],)+(1,)*(x.ndim-1))
                self.data[f"jet{jet}"]["mask"] = ~np.all(self.data[f"jet{jet}"][inputs[0]]==0, axis=-1, keepdims=True)
            elif noise_type == 'zero':
                self.logger.info(f"Replacing jet{jet} with zeros")
                for s in inputs:
                    self.data[f"jet{jet}"][s] = np.zeros_like(self.data[f"jet{jet}"][s])
                mask = self.data[f"jet{jet}"]["mask"]
                self.data[f"jet{jet}"]["mask"] = np.tile((np.mean(mask,axis=0,keepdims=True)>0.5).astype(mask.dtype), (mask.shape[0],)+(1,)*(mask.ndim-1))
            else:
                raise ValueError(f"Invalid noise type: {noise_type}")
        self.labels = np.zeros((num_events, 2),dtype=int)
        self.labels[~s_idx,0] = 1 #background labels
        self.labels[s_idx,1] = 1 #signal labels
        self.finish_init(batch_size, num_events//batch_size, num_events)

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        inputs = tuple(self.data[jet][s][start:end] for jet in self.jets for s in self.inputs)
        if(self.decoder_inputs): inputs = inputs + (self.data["decoder"][start:end],)
        return inputs, self.labels[start:end]

class MergedJetDataLoader(JetDataLoader):
    """
    Similar to JetDataLoader but merges both jets into one `jet`. Note that the resulting jet is not longer sorted by pT.
    """
    def __init__(self, bg, sn, batch_size:int, N_background:int, N_signal:int, particles:int=30, features=None, seed:int=1,
                 bg_slice:slice=slice(None), sn_slice:slice=slice(None),
                 inputs:list=["coords", "features", "mask"], decoder_inputs:list[DecoderFeature]=None, njets:int=2,discriminator:Discriminator=None, **kwargs):
        super().__init__(bg, sn, batch_size, N_background, N_signal, particles, features, seed,
                 bg_slice, sn_slice, inputs, decoder_inputs, njets, discriminator, **kwargs)
        for s in inputs:
            self.data[s] = np.concatenate([self.data[jet][s] for jet in self.jets], axis=1)
        for jet in self.jets:
            del self.data[jet]
    
    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        inputs = tuple(self.data[s][start:end] for s in self.inputs)
        if(self.decoder_inputs): inputs = inputs + (self.data["decoder"][start:end],)
        return inputs, self.labels[start:end]

class IADDataLoader(BaseDataLoader):
    """
    DEPRICATED, use IADDataLoaderV2 instead.
    Takes a background and signal dataset and combines them into two other datasets:
    1. Pure background (size: N_background)
    2. Background + some signal contamination (size: N_background (with N_signal events being replaced by signals))
    If `reuse_background=False`, the background dataset should contain at least 2*N_Background events-N_signal.
    Otherwise, the same events are used for pure background and contaminated data

    - Supports multiple jets
    - Supports decoder inputs
    - Supports discriminator
    - The data is shuffled.
    """
    def __init__(self, bg:Dict[str, np.ndarray], sn:Dict[str, np.ndarray], batch_size:int, N_background:int, N_signal:int, features=None, seed:int=1,
                 inputs:list=["coords", "features", "mask"], decoder_inputs:list[callable]=None, njets:int=2,discriminator:Discriminator=None, reuse_background=False):
        """
        params
        ------
        bg: dict,
            Background data dictionary.
        sn: dict,
            Signal data dictionary.
        batch_size: int,
            Size of each batch.
        N_background: int,
            Number of background events to load.
        N_signal: int,
            Number of signal events to load.
        features: list or None, optional
            List of features to select. Default is None.
            If `None`, no features are selected, if 'all', all features are selected. Otherwise, the indices in the list are selected.
        seed: int, optional,
            Random seed for reproducibility. Default is 1.
        inputs: list, optional,
            List of input types to load. Default is ["coords", "features", "mask"].
        decoder_inputs: list of callables or None, optional,
            List of decoder input functions. Default is None.
        njets: int, optional,
            Number of jets > 1. Default is 2.
        discriminator: Discriminator or None, optional,
            Discriminator object to filter events. Default is None.
        reuse_background: bool, optional, depricated
            If `True`, the same background events are used for the pure background and the contaminated data. Default is False.
            This option is depricated since setting it to `True` generally doesn't make any sense.
        """
        super().__init__(seed)
        self.inputs = inputs.copy()
        self.njets = njets
        self.jets = [f"jet{idx}" for idx in range(1,1+njets)]
        N_data_bg = bg["signal"].shape[0]
        N_data_sn = sn["signal"].shape[0]
        if(features is None):
            self.inputs.remove("features")

        #do some checks
        N_background_needed = N_background if reuse_background else 2*N_background-N_signal
        assert(N_data_bg >= N_background_needed and N_data_sn >= N_signal), \
            "Dataset is smaller than number of events to be read from it"
        assert all([bg[jet][s].shape[1:] == sn[jet][s].shape[1:] for jet in self.jets for s in inputs]), \
            "Shapes of background and signal don't match"

        #apply the discriminator
        disc = discriminator if discriminator is not None else Discriminator("jet_features/-1", None, None)
        idx_bg = np.where(disc.apply(bg))[0]
        idx_sn = np.where(disc.apply(sn))[0]
        n_background_in_data = len(idx_bg)
        n_signal_in_data = len(idx_sn)
        if(discriminator):
            print(f"{n_background_in_data} background and {n_signal_in_data} signal events passed the discriminator")

        #find background and signal events
        assert n_background_in_data>=N_background_needed, "Not enough background events"
        assert n_signal_in_data>=N_signal, "Not enough signal events"
        if(N_background < 0): N_background = n_background_in_data//2
        if(N_signal < 0): N_signal = n_signal_in_data
        num_events = 2*N_background
        
        #initialize data
        self.data = {jet: {s:np.empty((num_events, *bg[jet][s].shape[1:]), dtype=np.float32) for s in inputs} for jet in self.jets}
        
        #select the required number of background events
        if(reuse_background):
            idx1 = np.sort(self.rnd.choice(idx_bg, size=N_background, replace=False))
            idx2 = np.sort(self.rnd.choice(idx_bg, size=N_background-N_signal, replace=False))
        else:
            idx = np.sort(self.rnd.choice(idx_bg, size=2*N_background-N_signal, replace=False))
            idx1, idx2 = idx[:N_background], idx[N_background:]
        for s in inputs:
            for jet in self.jets:
                d = np.array(bg[f"{jet}/{s}"])[idx1]
                assert not(np.any(np.isnan(d)) or np.any(np.isinf(d))), "There is a nan value in the data"
                self.data[jet][s][:N_background] = d #put the chosen events into the pure background spots
                d = np.array(bg[f"{jet}/{s}"])[idx2]
                assert not(np.any(np.isnan(d)) or np.any(np.isinf(d))), "There is a nan value in the data"
                self.data[jet][s][N_background:2*N_background-N_signal] = d #put the chosen events into the contaminated spots

        if decoder_inputs is not None:
            if(len(decoder_inputs)>1):
                self.data["decoder"] = np.concatenate((dec(self.data) for dec in decoder_inputs),axis=-1)
            else:
                self.data["decoder"] = decoder_inputs[0](self.data)
            assert self.data["decoder"].shape[0] == num_events and self.data["decoder"].ndim == 2, \
                f"Decoder features do not have the right shape. Expected ({num_events}, n), got {self.data['decoder'].shape}"
            self.decoder_inputs = True
        else:
            self.decoder_inputs = False

        #select the required number of signal events
        idx = np.sort(self.rnd.choice(idx_sn, size=N_signal, replace=False))
        for s in inputs:
            for jet in self.jets:
                self.data[jet][s][-N_signal:] = np.array(sn[f"{jet}/{s}"])[idx] #put the chosen events into the contaminated spots

        if(features != 'all' and features is not None):
            #select the required features
            for jet in self.jets: self.data[jet]["features"] = self.data[jet]["features"][:,:,features] #yes, terrible code. I know
        self.labels = np.zeros((num_events, 2),dtype=int)
        self.true_labels = np.zeros_like(self.labels)
        self.labels[:N_background,0] = 1 #background labels
        self.labels[N_background:,1] = 1 #contaminated labels
        self.true_labels[:-N_signal,0] = 1 # true background
        self.true_labels[-N_signal:,1] = 1 # true signal
        print("Number of signal events:", np.sum(self.labels[:,1]), "of", self.labels.shape[0])
        print("Number of true signal events:", np.sum(self.true_labels[:,1]), "of", self.true_labels.shape[0])
        self.finish_init(batch_size, num_events//batch_size, num_events)
        self.shuffle()

    def shuffle(self):
        idx = super().shuffle()
        self.true_labels = self.true_labels[idx]
        iad_sums = np.sum(self.labels, axis=0)
        true_sums = np.sum(self.true_labels, axis=0)
        print(f"IAD Labels:  {iad_sums[0]:d} CR, {iad_sums[1]:d} SR")
        print(f"True Labels: {true_sums[0]:d} background, {true_sums[1]:d} signal")
        print(f"Signal in SR: {np.sum(self.true_labels[self.labels[:,1]==1,1]):d}")
        print(f"Signal in CR: {np.sum(self.true_labels[self.labels[:,1]==0,1]):d}")
    
    def get_true_labels(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        return self.true_labels[start:end]

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        inputs = tuple(self.data[jet][s][start:end] for jet in self.jets for s in self.inputs)
        if(self.decoder_inputs): inputs = inputs + (self.data["decoder"][start:end],)
        return inputs, self.labels[start:end]

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

def _select_indices(allowed_indices, num, rng):
    if(np.size(allowed_indices)==num):
        return slice(None) #selects all events
    return np.sort(rng.choice(allowed_indices, size=num, replace=False))

class IADDataLoaderV2(BaseDataLoader):
    """
    Replaces IADDataLoader.

    Takes a background and signal dataset and combines them into two other datasets:
    1. Pure background (size: N_background)
    2. Background + some signal contamination (size: N_background+N_signal)
    Note, that the background dataset should contain at least 2*N_Background events.

    - Supports different simulated background and data background
    - Supports multiple jets
    - Supports decoder inputs
    - Supports discriminator
    - Supports option to first take a slice from the data before selecting the events according to `N_background` and `N_signal`
        Useful for splitting the data into training and validation sets.
    - Supports oversampling
    - Supports adding noise to the features
    - Supports including the true labels in the output (for validation/testing)
    """
    def __init__(self, sim_bg, data_sn, data_bg, batch_size:int, N_simulated:int, N_background:int, N_signal:int, features=None, seed:int=1,
                 inputs:list=["coords", "features", "mask"], decoder_inputs:list[callable]=None, njets:int=2,discriminator:Discriminator=None, particles:int=30,
                 sim_bg_slice:list|slice=None, data_sn_slice:list|slice=None, data_bg_slice:list|slice=None, oversampling:str=None,
                 noise_features:int=None, noise_param:tuple=(0,1), noise_type:str='normal', include_true_labels:bool=False, do_shuffle:bool=True):
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
        inputs: list, optional,
            List of input types to load. Default is ["coords", "features", "mask"].
        decoder_inputs: list of callables or None, optional,
            List of decoder input functions. Default is None.
        njets: int, optional,
            Number of jets > 1. Default is 2.
        discriminator: Discriminator or None, optional,
            Discriminator object to filter events. Default is None.
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
        noise_type: str, optional,
            Type of noise distribution. Default is 'normal'.
        include_true_labels: bool, optional,
            Include true labels in the output. Default is False.
            Should be used for the validation set.
        do_shuffle: bool, optional,
            Shuffle the data. Default is True.
        """
        super().__init__(seed)
        self.inputs = inputs.copy()
        self.njets = njets
        self.jets = [f"jet{idx}" for idx in range(1,1+njets)] if njets is not None else [""]
        self.include_true_labels = include_true_labels
        
        if(features is None):
            self.inputs.remove("features")
        sim_bg_slice  = _check_slices(sim_bg_slice)
        data_sn_slice = _check_slices(data_sn_slice)
        data_bg_slice = _check_slices(data_bg_slice)

        #do some checks
        s = inputs[0] if njets is None else "jet1/"+inputs[0]
        assert(data_bg[s].shape[0] >= N_background and data_sn[s].shape[0] >= N_signal and sim_bg[s].shape[0] >= N_simulated), \
            "Dataset is smaller than number of events to be read from it"
        assert all([data_bg[jet][s].shape[2:] == data_sn[jet][s].shape[2:] for jet in self.jets for s in inputs]), \
            "Shapes of background and signal don't match"
        assert all([data_bg[jet][s].shape[1] >= particles for jet in self.jets for s in inputs]), \
            "Not enough particles per event (background)"
        assert all([data_sn[jet][s].shape[1] >= particles for jet in self.jets for s in inputs]), \
            "Not enough particles per event (signal)"
        # n_signals per batch contains a list of numpy arrays of size (100,) with each index's value corresponding to
        # the number of batches with a number of signal events equal to that index. ([0,2,1]) means 2 batches with 1 signal, 1 batches with 2 signals
        self.n_signal_per_batch = [] 

        #apply the discriminator
        if(discriminator is not None):
            passed_sim_bg_disc = discriminator.apply(sim_bg) #boolean array, true<=>event passed discriminator
            idx_sim = np.where(np.concatenate([passed_sim_bg_disc[s] for s in sim_bg_slice],axis=0))[0]
            passed_data_bg_disc = discriminator.apply(data_bg)
            idx_bg = np.where(np.concatenate([passed_data_bg_disc[s] for s in data_bg_slice],axis=0))[0]
            passed_data_sn_disc = discriminator.apply(data_sn)
            idx_sn = np.where(np.concatenate([passed_data_sn_disc[s] for s in data_sn_slice],axis=0))[0]
        else:
            idx_sim = np.concatenate((
                np.arange(_slice.start or 0, _slice.stop or sim_bg[s].shape[0], _slice.step or 1)-(_slice.start or 0)
                for _slice in sim_bg_slice), axis=0)
            idx_bg = np.concatenate((
                np.arange(_slice.start or 0, _slice.stop or data_bg[s].shape[0], _slice.step or 1)-(_slice.start or 0)
                for _slice in data_bg_slice),axis=0)
            idx_sn = np.concatenate((
                np.arange(_slice.start or 0, _slice.stop or data_sn[s].shape[0], _slice.step or 1)-(_slice.start or 0)
                for _slice in data_sn_slice),axis=0)
            
        n_simulated_in_data = len(idx_sim)
        n_background_in_data = len(idx_bg)
        n_signal_in_data = len(idx_sn)
        print(f"{n_simulated_in_data} simulated, {n_background_in_data} background and {n_signal_in_data} signal events passed the discriminator and/or slice")

        #find background and signal events
        assert n_simulated_in_data>=N_simulated, "Not enough background events"
        assert n_background_in_data>=N_background, "Not enough background events"
        assert n_signal_in_data>=N_signal, "Not enough signal events"
        if(N_simulated < 0): N_simulated = n_simulated_in_data
        if(N_background < 0): N_background = n_background_in_data
        if(N_signal < 0): N_signal = n_signal_in_data
        num_events = N_simulated+N_background+N_signal
        #initialize data
        self.data = {jet: {s:np.empty((num_events, particles, *sim_bg[jet][s].shape[2:]), dtype=np.float32) for s in inputs} for jet in self.jets}
        
        def no_nans(key, data):
            if(key == "features" and features != 'all'):
                return not(np.any(np.isnan(data[:,:,features])) or np.any(np.isinf(data[:,:,features])))
            else:
                return not(np.any(np.isnan(data)) or np.any(np.isinf(data)))

        #select the required number of simulated events
        idx = _select_indices(idx_sim, N_simulated, self.rnd)
        for s in inputs:
            for jet in self.jets:
                d = np.concatenate([sim_bg[f"{jet}/{s}"][_slice, :particles] for _slice in sim_bg_slice])[idx]
                assert no_nans(s,d), f"There is a nan value in the data ({jet}/{s})"
                self.data[jet][s][:N_simulated] = d #put the chosen events into the simulated spots (pure background)

        #select the required number of background events
        idx = _select_indices(idx_bg, N_background, self.rnd)
        for s in inputs:
            for jet in self.jets:
                d = np.concatenate([data_bg[f"{jet}/{s}"][_slice, :particles] for _slice in data_bg_slice])[idx]
                assert no_nans(s,d), f"There is a nan value in the data ({jet}/{s})"
                self.data[jet][s][N_simulated:N_simulated+N_background] = d #put the chosen events into the data background spots

        #select the required number of signal events
        idx = _select_indices(idx_sn, N_signal, self.rnd)
        for s in inputs:
            for jet in self.jets:
                d = np.concatenate([data_sn[f"{jet}/{s}"][_slice, :particles] for _slice in data_sn_slice])[idx]
                assert no_nans(s,d), f"There is a nan value in the data ({jet}/{s})"
                self.data[jet][s][N_simulated+N_background:] = d #put the chosen events into the data signal spots
        
        if decoder_inputs is not None:
            if(len(decoder_inputs)>1):
                self.data["decoder"] = np.concatenate((dec(self.data) for dec in decoder_inputs),axis=-1)
            else:
                self.data["decoder"] = decoder_inputs[0](self.data)
            assert self.data["decoder"].shape[0] == num_events and self.data["decoder"].ndim == 2, \
                f"Decoder features do not have the right shape. Expected ({num_events}, n), got {self.data['decoder'].shape}"
            self.decoder_inputs = True
        else:
            self.decoder_inputs = False

        if(features != 'all' and features is not None):
            #select the required features
            for jet in self.jets: self.data[jet]["features"] = self.data[jet]["features"][:,:,features] #yes, not very efficient. I know
        if(noise_features):
            from utils import noise_gen
            for jet in self.jets: self.data[jet]["features"] = np.concatenate((self.data[jet]["features"], noise_gen.sample(noise_type, noise_param,size=(*self.data[jet]["features"].shape[:-1],noise_features), seed=seed)), axis=-1)

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
                for s in inputs:
                    for jet in self.jets:
                        d = self.data[jet][s][N_simulated:]
                        self.data[jet][s] = np.concatenate((self.data[jet][s][:N_simulated], d[idx]),axis=0)
                self.labels = np.concatenate((self.labels[:N_simulated], self.labels[N_simulated:][idx]),axis=0)
                self.true_labels = np.concatenate((self.true_labels[:N_simulated], self.true_labels[N_simulated:][idx]),axis=0)
                if(self.decoder_inputs):
                    self.data["decoder"] = np.concatenate((self.data["decoder"][:N_simulated], self.data["decoder"][N_simulated:][idx]),axis=0)
            elif(N_simulated < N_data):
                #oversample simulated
                idx = oversample(N_simulated, N_data, mode=oversampling, shuffle=False, rng=self.rnd)
                for s in inputs:
                    for jet in self.jets:
                        d = self.data[jet][s][:N_simulated]
                        self.data[jet][s] = np.concatenate((d[idx], self.data[jet][s][N_simulated:]),axis=0)
                self.labels = np.concatenate((self.labels[:N_simulated][idx], self.labels[N_simulated:]),axis=0)
                self.true_labels = np.concatenate((self.true_labels[:N_simulated][idx], self.true_labels[N_simulated:]),axis=0)
                if(self.decoder_inputs):
                    self.data["decoder"] = np.concatenate((self.data["decoder"][:N_simulated][idx], self.data["decoder"][N_simulated:]),axis=0)

            num_events = self.labels.shape[0]

        self.finish_init(batch_size, num_events//batch_size, num_events)
        self.SR = N_signal/N_data
        if(do_shuffle):
            self.shuffle()

    def shuffle(self):
        idx = super().shuffle()
        self.true_labels = self.true_labels[idx]
        iad_sums = np.sum(self.labels, axis=0)
        true_sums = np.sum(self.true_labels, axis=0)
        print(f"IAD Labels:  {iad_sums[0]:d} CR, {iad_sums[1]:d} SR")
        print(f"True Labels: {true_sums[0]:d} background, {true_sums[1]:d} signal")
        print(f"Signal in SR: {np.sum(self.true_labels[self.labels[:,1]==1,1]):d}")
        print(f"Signal in CR: {np.sum(self.true_labels[self.labels[:,1]==0,1]):d}")
        self.n_signal_per_batch.append(np.zeros(int(self.batch_size*min(1,max(0.5,self.SR*1.5)))))
        self.max_signal_per_batch = self.n_signal_per_batch[-1].shape[0]-1
    
    def get_true_labels(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        return self.true_labels[start:end]

    def get_signals_per_batch(self, epoch=None):
        return self.n_signal_per_batch[epoch] if epoch else self.n_signal_per_batch

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        self.n_signal_per_batch[-1][min(np.sum(self.true_labels[start:end,1]),self.max_signal_per_batch)] += 1
        inputs = tuple(self.data[jet][s][start:end] for jet in self.jets for s in self.inputs)
        if(self.decoder_inputs): inputs = inputs + (self.data["decoder"][start:end],)
        if not self.include_true_labels:
            return inputs, self.labels[start:end]
        return inputs, (self.labels[start:end], self.true_labels[start:end])
    