from .DataLoader import DataLoader
from .Discriminator import Discriminator
import numpy as np
from time import time
from utils.misc import slice_to_indices
import utils.shuffling as shuffling

class JetDataLoader(DataLoader):
    def __init__(self, bg, sn, batch_size:int, N_background:int, N_signal:int, particles:int=30, features=None, seed:int=1,
                 bg_slice:slice=slice(None), sn_slice:slice=slice(None),
                 inputs:list=["coords", "features", "mask"], njets:int=2,discriminator:Discriminator=None):
        self.inputs = inputs.copy()
        self.stopped = False
        self.rnd = np.random.default_rng(seed)
        self.njets = njets
        self.jets = [f"jet{idx}" for idx in range(1,1+njets)]
        N_data_bg = bg["signal"][bg_slice].shape[0]
        N_data_sn = sn["signal"][sn_slice].shape[0]
        if(features is None):
            self.inputs.remove("features")

        tic = time()
        #do some checks
        assert(N_data_bg >= N_background and N_data_sn >= N_signal), \
            "Dataset is smaller than number of events to be read from it"
        assert all([bg[jet][s].shape[1:] == sn[jet][s].shape[1:] for jet in self.jets for s in inputs]), \
            "Shapes of background and signal don't match"
        assert bg[self.jets[0]][inputs[0]].shape[1] >= particles, \
            "Dataset does not contain enough particles per jet"
        
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
        self.N = N_background+N_signal

        s_idx = np.zeros(self.N, dtype=bool) #default: all background
        s_idx[self.rnd.choice(self.N, N_signal, replace=False)] = True #these will be signals
        #initialize data
        self.data = {jet: {s:np.empty((self.N, particles, *bg[jet][s].shape[2:]), dtype=np.float32) for s in inputs} for jet in self.jets}
        
        #select the required number of background events
        idx = np.sort(self.rnd.choice(idx_bg, size=N_background, replace=False))
        for s in inputs:
            for jet in self.jets:
                d = np.array(bg[f"{jet}/{s}"][bg_slice, :particles])[idx]
                assert not(np.any(np.isnan(d)) or np.any(np.isinf(d))), "There is a nan value in the data"
                self.data[jet][s][~s_idx] = d #put the chosen events into the background spots

        #select the required number of signal events
        idx = np.sort(self.rnd.choice(idx_sn, size=N_signal, replace=False))
        for s in inputs:
            for jet in self.jets:
                self.data[jet][s][s_idx] = np.array(sn[f"{jet}/{s}"][sn_slice,:particles])[idx] #put the chosen events into the signal spots

        if(features != 'all' and features is not None):
            #select the required features
            for jet in self.jets: self.data[jet]["features"] = self.data[jet]["features"][:,:,features] #yes, terrible code. I know
        self.labels = np.zeros((self.N, 2),dtype=int)
        self.labels[~s_idx,0] = 1 #background labels
        self.labels[s_idx,1] = 1 #signal labels
        self.N_batches = (self.N)//batch_size
        self.batch_size = batch_size
        self.time = time()-tic
        self.shuffle_time = 0
    
    def shuffle(self):
        tic = time()
        idx = np.arange(self.N)
        self.rnd.shuffle(idx)
        shuffling.do_the_shuffle(self.data, idx)
        shuffling.do_the_shuffle(self.labels, idx)
        self.shuffle_time += time()-tic
        return idx

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        return [[self.data[jet][s][start:end] for jet in self.jets for s in self.inputs], self.labels[start:end]]
    
class IADDataLoader(JetDataLoader):
    def __init__(self, bg, sn, batch_size:int, N_background:int, N_signal:int, features=None, seed:int=1,
                 inputs:list=["coords", "features", "mask"], njets:int=2,discriminator:Discriminator=None, reuse_background=False):
        """
        Takes a background and signal dataset and combines them into two other datasets:
        1. Pure background (size: N_background)
        2. Background + some signal contamination (size: N_background (with N_signal events being replaced by signals))
        If `reuse_background=False`, the background dataset should contain at least 2*N_Background events-N_signal.
        Otherwise, the same events are used for pure background and contaminated data
        """
        self.inputs = inputs.copy()
        self.stopped = False
        self.rnd = np.random.default_rng(seed)
        self.njets = njets
        self.jets = [f"jet{idx}" for idx in range(1,1+njets)]
        N_data_bg = bg["signal"].shape[0]
        N_data_sn = sn["signal"].shape[0]
        if(features is None):
            self.inputs.remove("features")

        tic = time()
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
        self.N = 2*N_background
        
        #initialize data
        self.data = {jet: {s:np.empty((self.N, *bg[jet][s].shape[1:]), dtype=np.float32) for s in inputs} for jet in self.jets}
        
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

        #select the required number of signal events
        idx = np.sort(self.rnd.choice(idx_sn, size=N_signal, replace=False))
        for s in inputs:
            for jet in self.jets:
                self.data[jet][s][-N_signal:] = np.array(sn[f"{jet}/{s}"])[idx] #put the chosen events into the contaminated spots

        if(features != 'all' and features is not None):
            #select the required features
            for jet in self.jets: self.data[jet]["features"] = self.data[jet]["features"][:,:,features] #yes, terrible code. I know
        self.labels = np.zeros((self.N, 2),dtype=int)
        self.true_labels = np.zeros_like(self.labels)
        self.labels[:N_background,0] = 1 #background labels
        self.labels[N_background:,1] = 1 #contaminated labels
        self.true_labels[:-N_signal,0] = 1 # true background
        self.true_labels[-N_signal:,1] = 1 # true signal
        print("Number of signal events:", np.sum(self.labels[:,1]), "of", self.labels.shape[0])
        print("Number of true signal events:", np.sum(self.true_labels[:,1]), "of", self.true_labels.shape[0])
        self.N_batches = (self.N)//batch_size
        self.batch_size = batch_size
        self.time = time()-tic
        self.shuffle_time = 0
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
    
class IADDataLoaderV2(IADDataLoader):
    def __init__(self, sim_bg, data_sn, data_bg, batch_size:int, N_simulated:int, N_background:int, N_signal:int, features=None, seed:int=1,
                 inputs:list=["coords", "features", "mask"], njets:int=2,discriminator:Discriminator=None,
                 sim_bg_slice=None, data_sn_slice=None, data_bg_slice=None, oversampling:str=None,
                 noise_features:int=None, noise_param:tuple=(0,1), noise_type:str='normal', include_true_labels:bool=False, do_shuffle:bool=True):
        """
        Takes a background and signal dataset and combines them into two other datasets:
        1. Pure background (size: N_background)
        2. Background + some signal contamination (size: N_background+N_signal)
        Note, that the background dataset should contain at least 2*N_Background events.
        """
        self.inputs = inputs.copy()
        self.stopped = False
        self.rnd = np.random.default_rng(seed)
        self.njets = njets
        self.jets = [f"jet{idx}" for idx in range(1,1+njets)] if njets is not None else [""]
        self.include_true_labels = include_true_labels
        
        if(features is None):
            self.inputs.remove("features")

        tic = time()
        #do some checks
        s = inputs[0] if njets is None else "jet1/"+inputs[0]
        assert(data_bg[s].shape[0] >= N_background and data_sn[s].shape[0] >= N_signal and sim_bg[s].shape[0] >= N_simulated), \
            "Dataset is smaller than number of events to be read from it"
        assert all([data_bg[jet][s].shape[1:] == data_sn[jet][s].shape[1:] for jet in self.jets for s in inputs]), \
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
        assert n_simulated_in_data>=N_simulated, "Not enough background events"
        assert n_background_in_data>=N_background, "Not enough background events"
        assert n_signal_in_data>=N_signal, "Not enough signal events"
        if(N_simulated < 0): N_simulated = n_simulated_in_data
        if(N_background < 0): N_background = n_background_in_data
        if(N_signal < 0): N_signal = n_signal_in_data
        self.N = N_simulated+N_background+N_signal
        #initialize data
        self.data = {jet: {s:np.empty((self.N, *sim_bg[jet][s].shape[1:]), dtype=np.float32) for s in inputs} for jet in self.jets}
        
        #select the required number of simulated events
        idx = np.sort(self.rnd.choice(idx_sim, size=N_simulated, replace=False))
        for s in inputs:
            for jet in self.jets:
                d = np.array(sim_bg[f"{jet}/{s}"][sim_bg_slice])[idx]
                assert not(np.any(np.isnan(d)) or np.any(np.isinf(d))), "There is a nan value in the data"
                self.data[jet][s][:N_simulated] = d #put the chosen events into the simulated spots (pure background)

        #select the required number of simulated events
        idx = np.sort(self.rnd.choice(idx_bg, size=N_background, replace=False))
        for s in inputs:
            for jet in self.jets:
                d = np.array(data_bg[f"{jet}/{s}"][data_bg_slice])[idx]
                assert not(np.any(np.isnan(d)) or np.any(np.isinf(d))), "There is a nan value in the data"
                self.data[jet][s][N_simulated:N_simulated+N_background] = d #put the chosen events into the data background spots

        #select the required number of signal events
        idx = np.sort(self.rnd.choice(idx_sn, size=N_signal, replace=False))
        for s in inputs:
            for jet in self.jets:
                d = np.array(data_sn[f"{jet}/{s}"][data_sn_slice])[idx]
                assert not(np.any(np.isnan(d)) or np.any(np.isinf(d))), "There is a nan value in the data"
                self.data[jet][s][N_simulated+N_background:] = d #put the chosen events into the data signal spots

        if(features != 'all' and features is not None):
            #select the required features
            for jet in self.jets: self.data[jet]["features"] = self.data[jet]["features"][:,:,features] #yes, not very efficient. I know
        if(noise_features):
            from utils import noise_gen
            for jet in self.jets: self.data[jet]["features"] = np.concatenate((self.data[jet]["features"], noise_gen.sample(noise_type, noise_param,size=(*self.data[jet]["features"].shape[:-1],noise_features), seed=seed)), axis=-1)

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
                for s in inputs:
                    for jet in self.jets:
                        d = self.data[jet][s][N_simulated:]
                        d = d[idx]
                        self.data[jet][s] = np.concatenate((self.data[jet][s][:N_simulated], d),axis=0)
                self.labels = np.concatenate((self.labels[:N_simulated], self.labels[N_simulated:][idx]),axis=0)
                self.true_labels = np.concatenate((self.true_labels[:N_simulated], self.true_labels[N_simulated:][idx]),axis=0)
            elif(N_simulated < N_data):
                #oversample simulated
                idx = oversample(N_simulated, N_data, mode=oversampling, shuffle=False, rng=self.rnd)
                for s in inputs:
                    for jet in self.jets:
                        d = self.data[jet][s][:N_simulated]
                        d = d[idx]
                        self.data[jet][s] = np.concatenate((d, self.data[jet][s][N_simulated:]),axis=0)
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

    def shuffle(self):
        super().shuffle()
        self.n_signal_per_batch.append(np.zeros(int(self.batch_size*min(1,max(0.5,self.SR*1.5)))))
        self.max_signal_per_batch = self.n_signal_per_batch[-1].shape[0]-1

    def get_signals_per_batch(self, epoch=None):
        return self.n_signal_per_batch[epoch] if epoch else self.n_signal_per_batch

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        self.n_signal_per_batch[-1][min(np.sum(self.true_labels[start:end,1]),self.max_signal_per_batch)] += 1
        if not self.include_true_labels:
            return [[self.data[jet][s][start:end] for jet in self.jets for s in self.inputs], self.labels[start:end]]
        return [[self.data[jet][s][start:end] for jet in self.jets for s in self.inputs], [self.labels[start:end], self.true_labels[start:end]]]
    