from .JetDataLoader import BaseDataLoader, _check_slices, _select_indices
from .Discriminator import Discriminator
import numpy as np
import numpy.typing as npt
from time import time
from typing import List
"""
NOTE: The IADDataLoaderV2 does not deal with k-fold crossvalidation, instead it supports lists of slices
    which can be used by the main script to select the correct folds with more flexibility. It can do this,
    because in the IAD setup, the CR and SR come from different datasets, to which k-fold crossvalivation can
    be applied independently.
    However, the CWoLaDataLoader takes care of distinguishing CR and SR itself, meaning that it also needs
    to take care of handling the k-fold cross-validation. Therefore, it does not support lists of slides for its
    datasets.
"""
class CWoLaDataLoader(BaseDataLoader):
    def __init__(self, background, signal, batch_size:int, SB_left:tuple, SR:tuple, SB_right:tuple, SR_feature_idx:int=-1,
                 N_background:int=-1, N_signal:int=-1,
                 features:List[int]=None, seed:int=1, inputs:List[str]=["coords", "features", "mask"], njets:int=2, particles:int=30,
                 background_slice:slice=None, signal_slice:slice=None, oversampling:str=None, noise_features:int=None, noise_param:tuple=(0,1),
                 cross_validation_params:dict=None, do_shuffle:bool=True, include_true_labels:bool=False):
        super().__init__(seed)
        self.inputs = inputs.copy()
        self.njets = njets
        self.jets = [f"jet{idx}" for idx in range(1,1+njets)]
        self.include_true_labels = include_true_labels

        N_data_bg = background[self.jets[0]]["mask"].shape[0]
        N_data_sn = background[self.jets[0]]["mask"].shape[0]

        assert(N_data_bg >= N_background and N_data_sn >= N_signal), \
            "Dataset is smaller than number of events to be read from it"
        assert all([background[jet][s].shape[1:] == signal[jet][s].shape[1:] for jet in self.jets for s in inputs]), \
            "Shapes of background and signal don't match"
        if not (N_background == -1 and N_signal == -1):
            raise DeprecationWarning("Specifiying `N_background` or `N_signal` is depricated, only values -1 are allowed")

        if(features is None):
            self.inputs.remove("features")

        disc_left = Discriminator("jet_features", col_idx=SR_feature_idx,lower=SB_left[0], upper=SB_left[1])
        disc_middle = Discriminator("jet_features", col_idx=SR_feature_idx,lower=SR[0], upper=SR[1])
        disc_right = Discriminator("jet_features", col_idx=SR_feature_idx,lower=SB_right[0], upper=SB_right[1])
        
        mask_disc_background_SB = np.logical_or(disc_left.apply(background, background_slice), disc_right.apply(background, background_slice))
        mask_disc_background_SR = disc_middle.apply(background, background_slice)
        n_background_in_SB, n_background_in_SR = np.sum(mask_disc_background_SB), np.sum(mask_disc_background_SR)
        
        mask_disc_signal_SB = np.logical_or(disc_left.apply(signal, signal_slice),disc_right.apply(signal, signal_slice))
        mask_disc_signal_SR = disc_middle.apply(signal, signal_slice)
        n_signal_in_SB, n_signal_in_SR     = np.sum(mask_disc_signal_SB), np.sum(mask_disc_signal_SR)
        print(f"Side bands:    {n_background_in_SB} background and {n_signal_in_SB} signal events passed the discriminator")
        print(f"Signal region: {n_background_in_SR} background and {n_signal_in_SR} signal events passed the discriminator")
        
        #now we know which events are allowed, so next we pick the correct fold
        #we do this by splittig array of allowed indices, rolling the splitted arrays based in the current cv index
        # and selecting the splits corresponding to the current train/val/test split
        k = cross_validation_params["K"]
        split = (np.array(cross_validation_params["split"])+cross_validation_params["index"])%k
        print("Cross validation split:", split)
        idx_background_SB_all, idx_background_SR_all = np.where(mask_disc_background_SB)[0], np.where(mask_disc_background_SR)[0]
        n_bg_SB, n_bg_SR = idx_background_SB_all.shape[0], idx_background_SR_all.shape[0]
        bg_SB_slices = np.array([slice(int(n_bg_SB/k*i), int(n_bg_SB/k*(i+1))) for i in range(k)])
        bg_SR_slices = np.array([slice(int(n_bg_SR/k*i), int(n_bg_SR/k*(i+1))) for i in range(k)])
        bg_SB_slices_split = [bg_SB_slices[idx] for idx in split]
        bg_SR_slices_split = [bg_SR_slices[idx] for idx in split]
        print("Using background SB slices: ", bg_SB_slices_split)
        print("Using background SR slices: ", bg_SR_slices_split)
        
        idx_background_SB = np.concatenate([idx_background_SB_all[s] for s in bg_SB_slices_split], axis=0)
        idx_background_SR = np.concatenate([idx_background_SR_all[s] for s in bg_SR_slices_split], axis=0)

        idx_signal_SB_all, idx_signal_SR_all = np.where(mask_disc_signal_SB)[0], np.where(mask_disc_signal_SR)[0]
        #Technically, we should have put the signal into the complete SR/SB datasets before doing the splitting into folds
        # in order to simulate that process, we assign each signal event in the SB/SR and index corresponding to a fold,
        # then we roll the folds by adding the fold index modulo k and take the events with the indices of the folds corresponding
        # to the current split
        fold_indices_SB = self.rnd.integers(0, k, size=idx_signal_SB_all.shape[0])
        idx_signal_SB = np.concatenate([idx_signal_SB_all[fold_indices_SB==i] for i in split],axis=0)
        fold_indices_SR = self.rnd.integers(0, k, size=idx_signal_SR_all.shape[0])
        idx_signal_SR = np.concatenate([idx_signal_SR_all[fold_indices_SR==i] for i in split],axis=0)
        print("SB using signal indices:", idx_signal_SB)
        print(f"Side bands:    {idx_background_SB.shape[0]} background and {idx_signal_SB.shape[0]} signal events in the {split} fold")
        print(f"Signal region: {idx_background_SR.shape[0]} background and {idx_signal_SR.shape[0]} signal events in the {split} fold")

        #calculate the total number of events
        N_background = idx_background_SB.shape[0] + idx_background_SR.shape[0]
        N_signal = idx_signal_SB.shape[0]+idx_signal_SR.shape[0]
        
        num_events = N_background+N_signal
        self.data = {jet: {s: np.empty((num_events, particles, *background[f"{jet}/{s:s}"].shape[2:]), dtype=np.float32) for s in inputs} for jet in self.jets}
        true_labels = np.zeros(num_events,dtype=np.int8)
        self.cwola_feature = np.empty(num_events)
        ##### General remark: DO NOT use any kind of fancy slicing on h5 files, that is abysmally slow.
        ##### If needed, first load the entire file into memory as a numpy array (or a chunk of it) and index later.
        def add_to_data(from_dataset, from_slice, to_slice, indices, label):
            for jet in self.jets:
                for s in inputs:
                    d = np.array(from_dataset[f"{jet}/{s}"][from_slice,:particles])[indices]
                    assert not(np.any(np.isnan(d)) or np.any(np.isinf(d))), "NaN or Inf in background data"
                    self.data[jet][s][to_slice] = d
                    true_labels[to_slice] = label
            self.cwola_feature[to_slice] = np.array(from_dataset["jet_features"][from_slice,SR_feature_idx])[indices]

        print(f"{background_slice=}, {N_background=}")
        add_to_data(background, background_slice, slice(0,N_background), np.concatenate((idx_background_SB, idx_background_SR),axis=0), 0)
        print(f"{signal_slice=}, {N_signal=}")
        add_to_data(signal, signal_slice, slice(N_background,None), np.concatenate((idx_signal_SB, idx_signal_SR),axis=0),1)

        if(features != 'all' and features is not None):
            #select the required features
            for jet in self.jets: self.data[jet]["features"] = self.data[jet]["features"][:,:,features] #yes, terrible code. I know
        if(noise_features):
            rng2 = np.random.default_rng(seed)
            for jet in self.jets: self.data["features"] = np.concatenate((self.data["features"], rng2.normal(*noise_param,size=(*self.data["features"].shape[:-1],noise_features))), axis=-1)

        #setup the true labels
        true_labels = np.stack((~true_labels, true_labels), axis=1,dtype=np.int8)
        #create cwola labels depending on whether the cwola feature (mjj) is in the side bands or in the signal region
        is_in_SR = np.logical_and(self.cwola_feature >= SR[0], self.cwola_feature < SR[1])
        cwola_labels = np.stack((~is_in_SR, is_in_SR),axis=1, dtype=np.int8)
        #do a sanity check
        test_labels = np.concatenate((np.zeros_like(idx_background_SB,dtype=bool), np.ones_like(idx_background_SR,dtype=bool),
                                    np.zeros_like(idx_signal_SB,dtype=bool), np.ones_like(idx_signal_SR,dtype=bool)),axis=0)
        assert np.all(test_labels==cwola_labels[:,1]), "Label sanity check failed!"
        N_SR = np.sum(is_in_SR)
        N_SB = num_events-N_SR
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
            num_events = true_labels.shape[0]

        self.labels = {"cwola": cwola_labels, "true": true_labels}
        self.finish_init(batch_size, num_events//batch_size, num_events)
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
            return tuple(self.data[jet][s][start:end] for jet in self.jets for s in self.inputs), (self.labels["cwola"][start:end], self.labels["true"][start:end])
        return tuple(self.data[jet][s][start:end] for jet in self.jets for s in self.inputs), self.labels["cwola"][start:end]
    
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