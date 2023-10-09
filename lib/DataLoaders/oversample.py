import numpy as np
from numpy import typing as npt
def oversample(dataset_size:int, num_samples:int, mode:str='choice', shuffle:bool=False,rng:np.random.Generator=None) -> npt.NDArray:
    """
    Samples `num_samples` from an array of size `dataset_size` (with `num_samples`>`dataset_size`) and returns the samples indices. 
    
    Parameters
    ----------
    dataset_size: int
        Size of the data to be samples
    num_samples: int
        Number of datapoints to be sampled
    mode: str, one of 'choice', 'repeat', 'repeat_strict', default: 'choice'
        `'choice'` samples using `numpy.random.choice`;
        `'repeat'` repeats the entire dataset as often as possible and uses `np.random.choice` (no replacement) for the remaining data points;
        `'repeat_script'` repeats the entire dataset as often as possible and ignores any remainer.
            Note, the latter will return only `dataset_size*(num_samples//dataset_size)` indices.
    shuffle: bool, default: False
        If `True`, the resulting indices will be shuffled
    rng: numpy.random.Generator, default: None
        A random number generator used to shuffle and sample. If none is provided, the default_rng is used.

    Returns
    -------
    An array of indices

    Example
    -------
    >>> arr = np.array([1,2,3,4])
    >>> idx = oversample(arr.shape[0], 10, 'repeat', False)
    >>> arr[idx]
    array([1, 2, 3, 4, 1, 2, 3, 4, 2, 3])
    """
    assert num_samples>dataset_size, "`num_samples` should be >`dataset_size`"
    if(rng is None):
        rng = np.random.default_rng()
    if(mode=='choice'):
        idx = rng.choice(dataset_size,num_samples, replace=True)
    elif(mode in ['repeat', 'repeat_strict']):
        repeats = num_samples//dataset_size
        t = [np.arange(dataset_size,dtype=np.int32)]*repeats
        if(mode=='repeat'):
            t += [rng.choice(dataset_size, num_samples-dataset_size*repeats, replace=False)]
        idx = np.concatenate(t,axis=0)
    if(shuffle):
        rng.shuffle(idx)
    return idx
