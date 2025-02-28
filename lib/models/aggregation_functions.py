# This file contains the aggregation functions used in PELICAN
import keras
from keras import ops as ko

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

def masked_mean(x:keras.KerasTensor,nobj:keras.KerasTensor, axis:int|tuple=None, keepdims:bool=False):
    x = ko.sum(x, axis=axis, keepdims=keepdims)
    if type(axis)!=int:
        nobj = nobj**(len(axis))
    nobj = ko.reshape(nobj, ([-1]+[1,]*(len(x.shape)-1)))
    return x/nobj

def masked_amax(x:keras.KerasTensor,nobj:keras.KerasTensor, axis:int|tuple=None, keepdims:bool=False):
    x = ko.max(x,axis=axis,keepdims=keepdims)
    return x - ko.log(nobj).reshape([-1],[1,]*(len(x.shape)-1))

def masked_amin(x:keras.KerasTensor,nobj:keras.KerasTensor, axis:int|tuple=None, keepdims:bool=False):
    x = ko.min(x,axis=axis,keepdims=keepdims)
    return x + ko.log(nobj).reshape([-1],[1,]*(len(x.shape)-1))

def masked_var(x:keras.KerasTensor,nobj:keras.KerasTensor, axis:int|tuple=None, keepdims:bool=False):
    return masked_mean((x-masked_mean(x, nobj, axis, keepdims=True))**2, nobj, axis, keepdims)
    
def masked_sum(x:keras.KerasTensor,nobj:keras.KerasTensor, axis:int|tuple=None, keepdims:bool=False):
    N = x.shape[-1]
    if type(axis)!=int:
        N = N**len(axis)
    return ko.sum(x, axis=axis, keepdims=keepdims)/N