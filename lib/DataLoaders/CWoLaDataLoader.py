from .JetDataLoader import JetDataLoader
from .Discriminator import Discriminator
import numpy as np
import numpy.typing as npt
from time import time

class CWoLaDataLoader(JetDataLoader):
    def __init__(self, background, signal, batch_size:int, N_background:int, N_signal:int, SB_left:tuple, SR:tuple, SB_right:tuple, SR_feature_idx:int=-1,
                 features=None, seed:int=1, inputs:list=["coords", "features", "mask"], njets:int=2, particles:int=30, do_shuffle:bool=True,
                 background_slice:slice=None, signal_slice:slice=None, oversampling:str=None, noise_features:int=None, noise_param:tuple=(0,1), include_true_labels:bool=False):
        self.inputs = inputs.copy()
        self.stopped = False
        self.rnd = np.random.default_rng(seed)
        self.njets = njets
        self.jets = [f"jet{idx}" for idx in range(1,1+njets)]
        self.include_true_labels = include_true_labels

        N_data_bg = background[self.jets[0]]["mask"].shape[0]
        N_data_sn = background[self.jets[0]]["mask"].shape[0]
        tic = time()

        assert(N_data_bg >= N_background and N_data_sn >= N_signal), \
            "Dataset is smaller than number of events to be read from it"
        assert all([background[jet][s].shape[1:] == signal[jet][s].shape[1:] for jet in self.jets for s in inputs]), \
            "Shapes of background and signal don't match"

        if(features is None):
            self.inputs.remove("features")

        disc_left = Discriminator("jet_features", col_idx=SR_feature_idx,lower=SB_left[0], upper=SB_left[1])
        disc_middle = Discriminator("jet_features", col_idx=SR_feature_idx,lower=SR[0], upper=SR[1])
        disc_right = Discriminator("jet_features", col_idx=SR_feature_idx,lower=SB_right[0], upper=SB_right[1])
        
        idx_disc_background_SB = np.logical_or(disc_left.apply(background, background_slice), disc_right.apply(background, background_slice))
        idx_disc_background_SR = disc_middle.apply(background, background_slice)
        idx_disc_background = np.logical_or(idx_disc_background_SB, idx_disc_background_SR)

        idx_disc_signal_SB = np.logical_or(disc_left.apply(signal, signal_slice),disc_right.apply(signal, signal_slice))
        idx_disc_signal_SR = disc_middle.apply(signal, signal_slice)
        idx_disc_signal = np.logical_or(idx_disc_signal_SB, idx_disc_signal_SR)

        n_background_in_data = np.sum(idx_disc_background)
        n_signal_in_data     = np.sum(idx_disc_signal)
        print(f"{n_background_in_data} background and {n_signal_in_data} signal events passed the discriminator")
        
        #find background and signal events
        assert n_background_in_data>=N_background, "Not enough background events"
        assert n_signal_in_data>=N_signal, "Not enough signal events"

        #pick the required number of background/signal events
        if N_background==-1: N_background = n_background_in_data
        if N_signal==-1: N_signal = n_signal_in_data
        
        self.N = N_background+N_signal
        self.data = {jet: {s: np.empty((self.N, particles, *background[f"{jet}/{s:s}"].shape[2:]), dtype=np.float32) for s in inputs} for jet in self.jets}
        true_labels = np.zeros(self.N,dtype=np.int8)
        self.cwola_feature = np.empty(self.N)
        ##### General remark: DO NOT use any kind of fancy slicing on h5 files, that is abysmally slow.
        ##### If needed, first load the entire file into memory as a numpy array (or a chunk of it) and index later
        def add_to_data(from_dataset, from_slice, to_slice, indices, label):
            for jet in self.jets:
                for s in inputs:
                    d = np.array(from_dataset[f"{jet}/{s}"][from_slice,:particles])[indices]
                    assert not(np.any(np.isnan(d)) or np.any(np.isinf(d))), "NaN or Inf in background data"
                    self.data[jet][s][to_slice] = d
                    true_labels[to_slice] = label
            self.cwola_feature[to_slice] = np.array(from_dataset["jet_features"][from_slice,SR_feature_idx])[indices]

        background_idx = self.rnd.choice(np.where(idx_disc_background)[0], N_background, replace=False)
        print(f"{background_slice=}, {N_background=}, {len(background_idx)}")
        add_to_data(background, background_slice, slice(0,N_background), background_idx, 0)
        signal_idx     = self.rnd.choice(np.where(idx_disc_signal)[0], N_signal, replace=False)
        print(f"{signal_slice=}, {N_signal=}, {len(signal_idx)}")
        add_to_data(signal, signal_slice, slice(N_background,None), signal_idx, 1)

        if(features != 'all' and features is not None):
            #select the required features
            for jet in self.jets: self.data[jet]["features"] = self.data[jet]["features"][:,:,features] #yes, terrible code. I know
        if(noise_features):
            rng2 = np.random.default_rng(seed)
            for jet in self.jets: self.data["features"] = np.concatenate((self.data["features"], rng2.normal(*noise_param,size=(*self.data["features"].shape[:-1],noise_features))), axis=-1)

        #setup the true labels
        true_labels = np.stack((1-true_labels, true_labels), axis=1)
        #create cwola labels depending on whether the cwola feature (mjj) is in the side bands or in the signal region
        is_in_SR = np.logical_and(self.cwola_feature >= SR[0], self.cwola_feature <= SR[1])
        cwola_labels = np.stack((1-is_in_SR, is_in_SR),axis=1)
        N_SR = np.sum(is_in_SR)
        N_SB = self.N-N_SR
        signal_in_SR = np.sum(np.logical_and(is_in_SR, true_labels[:,1]))
        print(f"{N_SR} events in SR region, of which {signal_in_SR} are signal")
        print(f"{N_SB} events in SB region, of which {np.sum(np.logical_and(~is_in_SR, true_labels[:,1]))} are signal")

        self.n_signal_per_batch = []
        self.signal_ratio = signal_in_SR/N_SR

        if(oversampling):
            from .oversample import oversample
            if(N_SB > N_SR):
                #oversample the signal region
                print(f"Oversampling signal region ({N_SR}) to match side bands ({N_SB}) using method '{oversampling}'")
                idx = oversample(N_SR, N_SB, mode=oversampling, shuffle=False, rng=self.rnd)
                for s in inputs:
                    for jet in self.jets:
                        d = self.data[jet][s][is_in_SR]
                        d = d[idx]
                        self.data[jet][s] = np.concatenate((self.data[jet][s][~is_in_SR], d),axis=0)
                cwola_labels = np.concatenate((cwola_labels[~is_in_SR], cwola_labels[is_in_SR][idx]),axis=0)
                true_labels = np.concatenate((true_labels[~is_in_SR], true_labels[is_in_SR][idx]),axis=0)
                self.cwola_feature = np.concatenate((self.cwola_feature[~is_in_SR], self.cwola_feature[is_in_SR][idx]), axis=0)
            elif(N_SB < N_SR):
                #oversample side bands
                print(f"Oversampling side bands ({N_SB}) to match signal region ({N_SR}) using method '{oversampling}'")
                idx = oversample(N_SB, N_SR, mode=oversampling, shuffle=False, rng=self.rnd)
                for s in inputs:
                    for jet in self.jets:
                        d = self.data[jet][s][~is_in_SR]
                        d = d[idx]
                        self.data[jet][s] = np.concatenate((d, self.data[jet][s][is_in_SR]),axis=0)
                cwola_labels = np.concatenate((cwola_labels[is_in_SR][idx], cwola_labels[~is_in_SR]),axis=0)
                true_labels = np.concatenate((true_labels[is_in_SR][idx], true_labels[~is_in_SR]),axis=0)
                self.cwola_feature = np.concatenate((self.cwola_feature[is_in_SR][idx], self.cwola_feature[~is_in_SR]), axis=0)
            self.N = true_labels.shape[0]

        self.labels = {"cwola": cwola_labels, "true": true_labels}
        self.N_batches = self.N//batch_size
        self.batch_size = batch_size
        self.time = time()-tic
        self.shuffle_time = 0
        if(do_shuffle):
            self.shuffle()
        
    def shuffle(self):
        idx = super().shuffle()
        tic = time()
        self.n_signal_per_batch.append(np.zeros(int(self.batch_size*min(1,max(0.5,self.signal_ratio*1.5)))))
        self.max_signal_per_batch = self.n_signal_per_batch[-1].shape[0]-1
        self.cwola_feature = self.cwola_feature[idx]
        self.shuffle_time += time()-tic

    def __getitem__(self, idx):
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        self.n_signal_per_batch[-1][min(np.sum(self.labels["true"][start:end,1]),self.max_signal_per_batch)] += 1
        if(self.include_true_labels):
            return [[self.data[jet][s][start:end] for jet in self.jets for s in self.inputs], [self.labels["cwola"][start:end], self.labels["true"][start:end]]]
        return [[self.data[jet][s][start:end] for jet in self.jets for s in self.inputs], self.labels["cwola"][start:end]]
    
    def get_true_labels(self,idx) -> npt.NDArray:
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        return self.labels["true"][start:end]

    def get_cwola_feature(self,idx) -> npt.NDArray:
        if(self.stopped):
            raise Exception("stop() has already been called!")
        start = idx*self.batch_size
        end = start+self.batch_size
        return self.cwola_feature[start:end]

    def get_signal_ratio(self):
        n_signal = np.sum(self.labels["cwola"][:,1])
        return n_signal/(self.labels["cwola"].shape[0]-n_signal)

    def get_signals_per_batch(self, epoch=None):
        return self.n_signal_per_batch[epoch] if epoch else self.n_signal_per_batch