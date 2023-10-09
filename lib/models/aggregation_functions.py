import tensorflow as tf
from tensorflow import keras

def get(name:str):
    if(name == "mean"):
        return masked_mean
    elif(name == "max"):
        return masked_amax
    elif(name == "min"):
        return masked_amin
    elif(name == "var"):
        return masked_var
    elif(name == "sum"):
        return masked_sum

def masked_mean(x:tf.Tensor,nobj:tf.Tensor, axis:int|tuple=None, keepdims:bool=False):
    x = tf.reduce_sum(x, axis=axis, keepdims=keepdims)
    if type(axis)!=int:
        nobj = nobj**(len(axis))
    nobj = tf.reshape(nobj, ([-1]+[1,]*(len(x.shape)-1)))
    return x/nobj

def masked_amax(x:tf.Tensor,nobj:tf.Tensor, axis:int|tuple=None, keepdims:bool=False):
    x = tf.reduce_max(x,axis=axis,keepdims=keepdims)
    return x - tf.math.log(nobj).reshape([-1],[1,]*(len(x.shape)-1))

def masked_amin(x:tf.Tensor,nobj:tf.Tensor, axis:int|tuple=None, keepdims:bool=False):
    x = tf.reduce_min(x,axis=axis,keepdims=keepdims)
    return x + tf.math.log(nobj).reshape([-1],[1,]*(len(x.shape)-1))

def masked_var(x:tf.Tensor,nobj:tf.Tensor, axis:int|tuple=None, keepdims:bool=False):
    return masked_mean((x-masked_mean(x, nobj, axis, keepdims=True))**2, nobj, axis, keepdims)
    
def masked_sum(x:tf.Tensor,nobj:tf.Tensor, axis:int|tuple=None, keepdims:bool=False):
    N = x.shape[-1]
    if type(axis)!=int:
        N = N**len(axis)
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)/N