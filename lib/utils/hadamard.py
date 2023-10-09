import numpy as np
import numpy.typing as npt

def hadamard(m:int, dtype:npt.DTypeLike=np.int16) -> npt.NDArray:
    if(m==0): return np.array([1], dtype=dtype)
    H_prev = hadamard(m-1)
    return np.block([[H_prev, H_prev],[H_prev, -H_prev]])

def part_eye(l:int,r:int,dtype:npt.DTypeLike=np.int16) -> npt.NDArray:
    if(l < r):
        return np.concatenate((np.eye(l, dtype=dtype), np.zeros((l,r-l), dtype=dtype)), axis=-1, dtype=dtype)
    if(l==r):
        return np.eye(l, dtype=dtype)
    if(l > r):
        return np.concatenate((np.eye(r, dtype=dtype), np.zeros((r,l-r), dtype=dtype)), axis=-1, dtype=dtype).T
    
def hadamard_weights(size:int|tuple, dtype:npt.DTypeLike=None) -> npt.NDArray:
    if(isinstance(size, np.number)):
        assert int(size) == size, f"Size should be an integer, got {size}"
        in_size = out_size = int(size)
    else:
        in_size, out_size = size
    m = np.ceil(np.log2(out_size))
    init = np.power(2, -(m-1)/2)*part_eye(in_size,int(2**m)) @ hadamard(m) @ part_eye(int(2**m), out_size)
    if(dtype is None):
        return init
    if(isinstance(dtype, type)):
        return init.astype(dtype=dtype)
    import tensorflow as tf
    return tf.convert_to_tensor(init, dtype=dtype)
        