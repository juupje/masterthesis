import numpy as np
import numpy.typing as npt

def do_the_shuffle(data:dict|list|npt.NDArray, idx:npt.NDArray):
    if(type(data) is np.ndarray):
        data[:] = data[idx]
        return
    iter = data.keys() if type(data) is dict else range(len(data))
    for k in iter:
        if(isinstance(data[k], (dict,list))):
            do_the_shuffle(data[k], idx)
        else:
            data[k] = data[k][idx]

def do_the_unshuffle(data:dict|list, idx:npt.NDArray):
    #idx is the shuffle index. To unshuffle, we 
    # sort the indices such that everything is back in order.
    idx = np.argsort(idx)
    do_the_shuffle(data,idx)