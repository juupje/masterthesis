import keras
from keras import ops as ko, KerasTensor
from typing import Tuple
import models.aggregation_functions as agg_fct
import re
from utils import activations
from numpy import sqrt

def eq2_to_0(x:KerasTensor, nparticles:KerasTensor, aggregation='mean') -> KerasTensor:
    """
    Calculates a full basis of aggregations of rank-2 tensors to rank-0 tensors (scalars).
    Parameters
    ----------
    `x`: Tensor(N,P,P,H)
    `nparticles`: Tensor(N,1,1)
    `aggregation`: type of aggregation function to use. See `aggregation_functions.py` for options (default: 'mean').
    
    Returns
    -------
    Tensor(N,H,2)
    """
    x = ko.transpose(x,axes=(0,3,1,2)) #(N,H,P,P)
    P = x.shape[-1] #number of particles
    diag_part = ko.diagonal(x, axis1=2, axis2=3) #(N,H,P)
    aggregation = agg_fct.get(aggregation)
    sum_diag_part = aggregation(diag_part,nparticles,axis=2) #(N,H)
    sum_all = aggregation(x, nparticles, axis=(2,3)) #(N,H)
    return ko.stack((sum_diag_part, sum_all), axis=2) #(N,H,2)

def eq2_to_2(x:KerasTensor, nparticles:KerasTensor, aggregation='mean', skip_order_zero=False) -> KerasTensor:
    """
    Calculates a full basis of aggregations of rank-2 tensors to rank-2 tensors.
    
    Parameters
    ----------
    `x`: Tensor(N,P,P,H)
    `nparticles`: Tensor(N,P,1)
    `aggregation`: type of aggregation function. See `aggregation_functions.py` for possibilities
    `skip_order_zero`: if True, the first 5 aggregation will be skipped

    Returns
    -------
    Tensor(N,H,15,P,P) if `skip_order_zero` is False, else Tensor(N,H,10,P,P)
    """
    x = ko.transpose(x, axes=(0,3,1,2)) #(N,H,P,P) (No idea why, but let's stick to the source)
    P = x.shape[-1] #number of particles, incl. zero padding
    H = x.shape[1] #number of particles, incl. zero padding
    diag_part = ko.diagonal(x,axis1=2,axis2=3) #(N,H,P)
    aggregation = agg_fct.get(aggregation)
    
    diag = ko.reshape(ko.eye(P), (1, 1, P, P)) #used for diagonalization
    def diagonalize(x:KerasTensor) -> KerasTensor:
        """
        Diagonalizes the last dimension of a tensor
        Parameters
        ----------
        `x`: Tensor(N,H,P)
        
        Returns
        -------
        Tensor(N,H,P,P)
        """
        return ko.expand_dims(x, axis=-1)*diag

    sum_diag_part = aggregation(diag_part, nparticles, axis=2, keepdims=True) #(N,H,1)
    sum_rows = aggregation(x, nparticles, axis=3) #(N,H,p)
    sum_cols = aggregation(x, nparticles, axis=2) #(N,H,p)
    sum_all = aggregation(x, nparticles, axis=(2,3)) #(N,H)

    ops = [None]*15
    # The reshapes are just here to make Keras know the shapes of the tensors
    # For some reason, ko.tile returns a [None,None,None ...] tensor
    ops[5]  = ko.reshape(ko.tile(ko.expand_dims(sum_cols, axis=2),(1,1,P,1)), (-1,H,P,P))
    ops[6]  = ko.reshape(ko.tile(ko.expand_dims(sum_rows, axis=2),(1,1,P,1)), (-1,H,P,P))
    ops[7]  = ko.reshape(ko.tile(ko.expand_dims(sum_diag_part, axis=3),(1,1,P,P)), (-1,H,P,P))
    ops[8]  = ko.reshape(ko.tile(ko.expand_dims(sum_cols, axis=3),(1,1,1,P)), (-1,H,P,P))
    ops[9]  = ko.reshape(ko.tile(ko.expand_dims(sum_rows, axis=3),(1,1,1,P)), (-1,H,P,P))
    ops[10] = diagonalize(ko.reshape(ko.tile(sum_diag_part,(1,1,P)), (-1,H,P)))
    ops[11] = diagonalize(sum_rows)
    ops[12] = diagonalize(sum_cols)
    ops[13] = ko.tile(ko.expand_dims(ko.expand_dims(sum_all, axis=-1),axis=-1),(1,1,P,P))
    ops[14] = diagonalize(ko.reshape(ko.tile(ko.expand_dims(sum_all,axis=-1), (1,1,P)), (-1,H,P)))
    if not skip_order_zero:
        ops[0] = x #(N,H,P,P)
        ops[1] = ko.transpose(x, axes=(0,1,3,2)) #(N,H,P,P)
        ops[2] = ko.tile(ko.expand_dims(diag_part, 2),(1,1,P,1)) #(N,H,P,P)
        ops[3] = ko.tile(ko.expand_dims(diag_part, 3),(1,1,1,P)) #(N,H,P,P)
        ops[4] = diagonalize(diag_part) #(N,H,P,P)
        return ko.stack(ops,axis=2)
    return ko.stack(ops[5:],axis=2)

class PelicanEmbedding(keras.layers.Layer):
    """
    Performs an embedding of the type `((1+x)^delta-1)/delta`,
    Where `delta` is a tensor of the same shape as `x` but a large last dimension.
    `delta`=`beta` where `beta` is a a linspace from 0.1 to 0.5. (This assumes that `x` is sorted)
    """
    def __init__(self, units=20, name="Embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    def build(self, input_shape:tuple):
        #Input shape = (N,...,1)
        outshape = [1]*(len(input_shape)-1)+[self.units]
        self.beta = keras.Variable(ko.reshape(ko.linspace(0.1,0.5,self.units),outshape), dtype='float32',
                trainable=True, name="beta")

    def call(self, x:KerasTensor) -> KerasTensor:
        delta = 1e-6+ko.square(self.beta)
        output = (ko.power(ko.abs(1+x), delta)-1)/delta
        return output

aggs = {"m":"mean", "s":"sum", "x": "max", "n":"min"}
class Eq2to0(keras.layers.Layer):
    """
    Implementation of an permutation equivariant 2->0 block
    Consist of a several full-sets of aggregations which are combined into an output using a linear layer
    """
    def __init__(self, units=50, activation_agg='leakyrelu', activation_lin=None, ir_safe=False,
                avg_nparticles=49, aggregations="M", name="Eq2to0", **kwargs):
        """
        Parameters
        ----------
        `units`: number of nodes in the linear layer (also output dimension)
        `activation_agg`: activation function applied after the aggregations (default: 'leakyrelu')
        `activation_lin`: activation function after the linear layer (default: None)
        `ir_safe`: if True, no biases are used (default: False)
        `avg_nparticles`: average number of particles per event (~50 for toptagging)
        `aggregations`: A string in which each letter represents a kind of aggregation function. Possible letters are `sxmn`.
                    Capital letters will also include a weighting of the different aggregations. More letters mean more aggregation blocks.
                    All aggregation blocks are combined into a single output using the linear layer. (Default: 'M')
        `name`: name of the layer.

        Returns
        -------
        Tensor(N,`units`)
        """
        super().__init__(name=name, **kwargs)
        self.activation_agg = activation_agg
        self.activation_lin = activation_lin
        self.ir_safe = ir_safe
        self.avg_nparticles = avg_nparticles
        assert re.fullmatch("^[mxnsMXNS]+$", aggregations)
        self.aggregations = aggregations
        self.out_dim = units #H2
        self.basis_dim = 2*len(self.aggregations) #D

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.out_dim, "activation_agg": self.activation_agg, "activation_lin": self.activation_lin,
                        "ir_safe": self.ir_safe, "avg_nparticles": self.avg_nparticles, "aggregations": self.aggregations})
        return config

    def build(self, input_shapes):
        self.in_dim = input_shapes[0][-1]

        self.alphas = [None]*len(self.aggregations)
        for idx, letter in enumerate(self.aggregations):
            if(letter.upper()==letter):
                self.alphas[idx] = self.add_weight(name=f"alpha{idx}",shape=(1,self.in_dim,2), dtype='float32',
                                trainable=True, initializer=keras.initializers.RandomUniform(0.,1.)) #(1,H,2)
        self.dense = keras.layers.Dense(self.out_dim, kernel_initializer=keras.initializers.HeNormal(),
                        activation=None, use_bias=not self.ir_safe)
        if(self.activation_lin is not None):
            self.activation_lin_layer = activations.get_activation(self.activation_lin)
        if(self.activation_agg is not None):
            self.activation_agg_layer = activations.get_activation(self.activation_agg)

    def call(self, inputs:Tuple[KerasTensor, KerasTensor]) -> KerasTensor:
        x, nparticles = inputs
        ops = []
        for idx, letter in enumerate(self.aggregations):
            if(letter.lower() in "smxn"):
                op = eq2_to_0(x,nparticles, aggregation=aggs[letter.lower()]) #(N,H,A)
                if(self.alphas[idx] is not None): #so, letter is a capital
                    mult = (ko.reshape(nparticles,(-1,1,1))**self.alphas[idx]) / (self.avg_nparticles**self.alphas[idx])
                    op = op*mult #(N,H,A)
                ops.append(op)
            else:
                raise ValueError(f"Invalid aggregation mode: '{letter}'")
        ops = ko.concatenate(ops,axis=2) #(N,H,D)
        if(self.activation_agg):
            ops = self.activation_agg_layer(ops)
        ops = ko.reshape(ops, (-1, ops.shape[-1]*ops.shape[-2])) #(N,H*D)
        output = self.dense(ops) #(N,H2)
        if(self.activation_lin):
            output = self.activation_lin_layer(output)
        return output #(N,H2)

class Eq2to2(keras.layers.Layer):
    """
    Implementation of an permutation equivariant 2->2 block
    Consist of a several full-sets of aggregations which are combined into an output using a linear layer
    """
    def __init__(self, units=50, activation_agg=None, activation_lin=None, ir_safe=False,
            avg_nparticles=49, aggregations="M", factorize=True, name="Eq2to2", **kwargs):
        """
        Parameters
        ----------
        `units`: number of nodes in the linear layer (also output dimension)
        `activation_agg`: activation function applied after the aggregations (default: None)
        `activation_lin`: activation function after the linear layer (default: None)
        `ir_safe`: if True, no biases are used (default: False)
        `avg_nparticles`: average number of particles per event (~50 for toptagging)
        `aggregations`: A string in which each letter represents a kind of aggregation function. Possible letters are `sxmn`.
                    Capital letters will also include a weighting of the different aggregations. More letters mean more aggregation blocks.
                    All aggregation blocks are combined into a single output using the linear layer. (Default: 'M')
        `name`: name of the layer.

        Returns
        -------
        Tensor(N,P,P,`units`)
        """
        super().__init__(name=name, **kwargs)
        self.out_dim = units #H2
        self.activation_agg = activation_agg
        self.activation_lin = activation_lin
        self.ir_safe = ir_safe
        self.avg_nparticles = avg_nparticles
        assert re.fullmatch("^[mxnsMXNS]+$", aggregations)
        self.aggregations = aggregations
        self.factorize = factorize
        self.basis_dim = 15+10*(len(self.aggregations)-1) #D

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.out_dim, "activation_agg": self.activation_agg, "activation_lin": self.activation_lin,
                        "ir_safe": self.ir_safe, "avg_nparticles": self.avg_nparticles, "aggregations": self.aggregations,
                        "factorize": self.factorize})
        return config

    def build(self, input_shapes):
        self.in_dim = input_shapes[0][-1] #H

        #Create the weights used to combine the 15 (or 10) different aggregations
        self.alphas = [None]*len(self.aggregations)
        for idx, letter in enumerate(self.aggregations):
            if(letter.upper()==letter):
                self.alphas[idx] = self.add_weight(name=f"alpha{idx}",shape=(1,self.in_dim,10,1,1), dtype='float32',
                                trainable=True, initializer=keras.initializers.RandomUniform(0.,1.)) #(1,H,10,1,1)
        self.dummy_alphas = ko.zeros((1,self.in_dim,5,1,1),dtype='float32') #(1,H,5,1,1)
        #create the dense layer
        if(self.factorize):
            self.coefs00 = keras.Variable(keras.random.normal((self.in_dim, self.basis_dim), 0, 1./sqrt(self.basis_dim), dtype='float32'),
                                trainable=True, name="coefs00")
            self.coefs01 = keras.Variable(keras.random.normal((self.basis_dim, self.out_dim), 0, 1./sqrt(self.basis_dim), dtype='float32'),
                                trainable=True, name="coefs01")
            self.coefs10 = keras.Variable(keras.random.normal((self.in_dim, self.out_dim), 0, 1./sqrt(self.in_dim), dtype='float32'),
                                trainable=True, name="coefs10")
            self.coefs11 = keras.Variable(keras.random.normal((self.in_dim, self.out_dim), 0, 1./sqrt(self.in_dim), dtype='float32'),
                                trainable=True, name="coefs11")
        else:
            self.dense = keras.layers.Dense(self.out_dim, kernel_initializer=keras.initializers.HeNormal(), activation=None, use_bias=None)
        
        if(self.activation_lin is not None):
            self.activation_lin_layer = activations.get_activation(self.activation_lin)
        if(self.activation_agg is not None):
            self.activation_agg_layer = activations.get_activation(self.activation_agg)
        #Also, calculate the biases
        if not self.ir_safe:
            self.bias = keras.Variable(ko.zeros((1,1,1,self.out_dim),dtype='float32'), trainable=True, name='bias') #(1,1,1,H2)
            self.diag_bias = keras.Variable(ko.zeros((1,1,1,self.out_dim),dtype='float32'), trainable=True, name='diag_bias') #(1,1,1,H2)
            self.eye = ko.expand_dims(ko.expand_dims(ko.eye(input_shapes[0][1]),axis=0),axis=-1) #(1,P,P,1)
   
    def call(self, inputs:Tuple[KerasTensor, KerasTensor, KerasTensor]) -> KerasTensor:
        x, edge_mask, nparticles = inputs
        # x = (N,P,P,H)
        # nparticles = (N,1,1)
        ops = []
        #Loop through the different aggregation functions and create 15 (or 10) aggregations for all of them
        for idx, letter in enumerate(self.aggregations):
            if(letter.lower() in "smxn"):
                op = eq2_to_2(x, nparticles, aggregation=aggs[letter.lower()], skip_order_zero=(idx!=0)) #(N,H,A,P,P) H=indim, A=15 or 10
                if(self.alphas[idx] is not None): #meaning, `letter` is capital -> do weighting
                    alphas = ko.concatenate((self.dummy_alphas, self.alphas[0]),axis=2) if idx==0 else self.alphas[idx] #(1,H,A,1,1)
                    mult = (ko.reshape(nparticles,(-1,1,1,1,1))**alphas)/(self.avg_nparticles**alphas)
                    op = op*mult #(N,H,A,P,P)
            else:
                raise ValueError(f"Invalid aggregation function: '{letter}'")
            ops.append(op)

        #Concatenate all the aggregation results and apply an activation
        ops = ko.concatenate(ops,axis=2) #(N,H,D,P,P), D=self.basis_dim
        if(self.activation_agg):
            ops = self.activation_agg_layer(ops)
        
        #Apply coefficients to each feature and aggregation and sum over them
        ops = ko.transpose(ops, axes=(0,3,4,1,2)) #(N,P,P,H,D)
        ops = ko.reshape(ops, (-1,)+tuple(ops.shape[1:-2])+(ops.shape[-2]*ops.shape[-1],)) #flatten last two layers -> (N,P,P,H*D)
        if(self.factorize):
            #                      (H,D,1)                     (H,1,H2)                                         (1,D,H2)                                (H,1,H2)
            coefs = ko.expand_dims(self.coefs00, axis=-1)*ko.expand_dims(self.coefs10, axis=1)*ko.expand_dims(self.coefs01, axis=0)*ko.expand_dims(self.coefs11, axis=1)
            #coefs = (in_dim, basis_dim, out_dim)
            coefs = ko.reshape(coefs,(-1,self.out_dim))
            output = ko.matmul(ops, coefs)
        else:
            output = self.dense(ops)
        if not self.ir_safe:
            diag_bias = self.eye*self.diag_bias
            output = output+self.bias+diag_bias
        #Apply another activation function
        if self.activation_lin:
            output = self.activation_lin_layer(output)
        output =  ko.where(edge_mask, output, 0)
        return output