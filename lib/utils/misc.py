import os, json
import numpy as np

def get_run_id(file:str,inc:bool=True) -> int:
    if(os.path.exists(file)):
        with open(file, 'r') as infile:
            doc = json.load(infile)
            run_id = doc["run_id"]
    else:
        doc = {}
        run_id = 1
    if(inc):
        doc["run_id"] = run_id + 1
        with open(file, "w") as outfile:
            json.dump(doc, outfile)
    return run_id
    
def slice_to_logical_indexing(slice, shape):
    size = shape[0] if isinstance(shape, (tuple,list)) else shape
    idx = np.zeros(size, dtype=bool)
    idx[slice] = True
    return idx

def clamp(x, low, high):
    return min(max(x,low),high)

def slice_to_indices(_slice, shape):
    if(type(shape) is int): shape = (shape,)
    if(not isinstance(_slice,(list,tuple))):
        _slice = (_slice, *(None,)*(len(shape)-1))
    assert len(_slice)==len(shape), "dimension of _slice and shape not compatible"
    indices = []
    for s, dim in zip(_slice,shape):
        if(s is None):
            indices.append(np.arange(0,dim))
        else:
            start,stop,step = s.start, s.stop, s.step
            if(start is None): start = 0 if (step or 1)>0 else dim
            if(stop is None): stop = dim if (step or 1)>0 else 0
            if(step is None): step = 1
            if(stop < 0): stop = dim-stop
            if(start < 0): start = dim-start
            start = clamp(start,0,dim-1)
            stop = clamp(stop, 0, dim)
            indices.append(np.arange(start,stop,step))
    dims = [i.shape[0] for i in indices]
    print(dims)
    for i in range(len(indices)):
        indices[i] = np.tile(np.repeat(indices[i],np.prod(dims[i+1:])), np.prod(dims[:i]))
    return tuple(indices), dims

def restore_ordering(pred, labels, data_idx):
    is_bg = labels==0
    order = np.argsort(data_idx[is_bg])
    pred_bg, labels_bg = pred[is_bg][order], labels[is_bg][order]
    order = np.argsort(data_idx[~is_bg])
    pred_sn, labels_sn = pred[~is_bg][order], labels[~is_bg][order]
    return np.concatenate((pred_bg, pred_sn)), np.concatenate((labels_bg, labels_sn))