import numpy as np
import h5py
from tensorflow import keras
from time import time

class DataGenerator(keras.utils.Sequence):
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
        assert all([bg[s].shape[0] == N_data_sn for s in inputs]), \
            "Not all inputs have the same number of rows"
        
        self.N = N_background+N_signal
        if(features is None):
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
        tic = time()
        idx = np.arange(self.N)
        self.rnd.shuffle(idx)
        for s in self.inputs: self.data[s] = self.data[s][idx]
        self.labels = self.labels[idx]
        self.shuffle_time += time()-tic
    
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

class SplitDataGenerator(keras.utils.Sequence):
    def __init__(self, bg, sn, batch_size, N_background, N_signal, seed, reshuffle=False, inputs=["coords", "features", "mask"]):
        self.inputs = inputs
        if(bg[inputs[0]].shape[0] < N_background or sn[inputs[0]].shape[0] < N_signal):
            raise ValueError("Dataset is smaller than number of events to be read from it")
        if(bg[inputs[0]].shape!=sn[inputs[0]].shape):
            raise ValueError("Shapes of coordinates of jets don't match")
        self.bg = {key: np.array(bg[key], dtype=np.float32) for key in inputs}
        self.sn = {key: np.array(sn[key], dtype=np.float32) for key in inputs}
        self.sample_count = N_background+N_signal
        self.N_bg = N_background
        self.N_sn = N_signal
        self.seed = seed
        ratio = N_signal/self.sample_count
        self.sn_batch_size = int(batch_size*ratio)
        self.bg_batch_size = batch_size-self.sn_batch_size
        self.batch_size = batch_size
        self.epoch = 0
        self.time = 0
        self.shuffle_time = 0
        self.N_batches = int(np.floor((self.sample_count) / self.batch_size))
        self.reshuffle = reshuffle
        self._reset()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.batches_list)
  
    def on_epoch_end(self):
        self.epoch += 1
        self._reset()

    def get_time_used(self):
        return self.time
    def get_time_used_shuffle(self):
        return self.shuffle_time

    def __getitem__(self, index):
        return self.batches[self.batches_list[index]]

    def _construct_batch(self, batch_number):
        start_bg     = batch_number * self.bg_batch_size
        end_bg       = min(start_bg + self.bg_batch_size, self.N_bg)
        start_sn     = batch_number * self.sn_batch_size
        end_sn       = min(start_sn + self.sn_batch_size, self.N_sn)
        # Load data from disk
        x = []
        for s in self.inputs:
            x.append(np.concatenate((self.bg[s][start_bg:end_bg], self.sn[s][start_sn:end_sn]), axis=0))
        y = np.concatenate((np.repeat([[1,0]], end_bg-start_bg, axis=0), np.repeat([[0,1]], end_sn-start_sn, axis=0)), axis=0)

        if(end_bg-start_bg != self.bg_batch_size):
            idx = np.sort(np.random.choice(start_bg,self.bg_batch_size-(end_bg-start_bg), replace=False))
            for i,s in enumerate(self.inputs):
                x[i] = np.concatenate((x[i],self.bg[s][idx]), axis=0)
            y = np.concatenate((y,np.repeat([[1,0]], self.bg_batch_size-(end_bg-start_bg), axis=0)), axis=0)

        if(end_sn-start_sn != self.sn_batch_size):
            idx = np.sort(np.random.choice(start_sn,self.sn_batch_size-(end_sn-start_sn), replace=False))
            for i,s in enumerate(self.inputs):
                x[i] = np.concatenate((x[i],self.sb[s][idx]), axis=0)
            y = np.concatenate((y,np.repeat([[0,1]], self.sn_batch_size-(end_sn-start_sn), axis=0)), axis=0)
        return x, y

    def _shuffle_data(self):
        tic = time()
        for x, nrows in zip((self.bg, self.sn), (self.N_bg, self.N_sn)):
            buf_size = nrows #min(10000, nrows)
            idx = np.arange(buf_size)
            np.random.shuffle(idx)
            for s in self.inputs:
                buf = x[s][idx]
                x[s] = buf
        self.shuffle_time += time()-tic

    def _construct_batches(self):
        self.batches = []
        tic = time()
        for batch_number in range(self.N_batches):
            self.batches.append(self._construct_batch(batch_number))
        self.time += time()-tic

    def _reset(self):
        if(self.reshuffle or not hasattr(self, 'batches')):
            #create the batches only once
            if(self.reshuffle or self.epoch==0):
                self._shuffle_data()
            self._construct_batches()
        self.batches_list = np.arange(self.N_batches,dtype=int)
        np.random.shuffle(self.batches_list)
        while(self.bg_batch_size*len(self.batches_list) > self.N_bg or
                self.sn_batch_size*len(self.batches_list) > self.N_sn):
            #stupid edge cases due to rounding
            self.batches_list = self.batches_list[:-1]
    
class BatchDataGenerator(SplitDataGenerator):
    #Override
    def __init__(self, bg, sn, batch_size, N_background, N_signal, seed, reshuffle=False):
        raise DeprecationWarning("This should not be used, since it cannot shuffle data")
        from tempfile import TemporaryFile
        import h5py
        self.temp_file = TemporaryFile(dir=os.path.join(os.getenv("TMP")))
        self.batch_file = h5py.File(self.temp_file, 'w')
        super().__init__(bg, sn, batch_size, N_background, N_signal, seed, reshuffle)

    #Override
    def __getitem__(self, index):
        #read one batch of data
        if(index>len(self.batches_list)):
            print("that shouldn't happen")
        tic = time()
        idx = self.batches_list[index]
        x = [np.array(self.batch_file[s][idx]) for s in self.inputs]
        y = np.array(self.batch_file["labels"][idx])
        self.time += time()-tic
        return x, y

    #Override
    def _construct_batches(self):
        #creates a temporary h5 file in which the batches are stored.
        # they can then be read when required for the training
        print("Constructing batches!")
        tic = time()
        if("labels" not in self.batch_file):
            #create the datasets
            for s in self.inputs:
                self.batch_file.create_dataset(s, shape=(self.N_batches, self.batch_size, *self.bg[s].shape[1:]), dtype=np.float32)
            self.batch_file.create_dataset("labels", shape=(self.N_batches, self.batch_size, 2), dtype=np.float32)
        for batch_number in range(self.N_batches):
            x, y = self._construct_batch(batch_number)
            for i in range(len(x)):
                self.batch_file[self.inputs[i]][batch_number,...] = x[i]
            self.batch_file["labels"][batch_number,...] = y
        self.time += time()-tic
    
    def __del__(self):
        if hasattr(self, 'batch_file'):
            self.batch_file.close()
        if hasattr(self, 'temp_file'):
            self.temp_file.close()
