import keras
from keras import ops as ko
from keras import KerasTensor
import tensorflow as tf

class PrintShape(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, *args, **kwargs):
        return tuple()
    
    def call(self, x):
        tf.print(f"Shape of {x.name}:",ko.shape(x))

class AdjacencyMatrix(keras.Layer):
    def __init__(self, n_nodes, include_segments=False, **kwargs):
        super().__init__(**kwargs)
        self.n_nodes = n_nodes
        self.include_segments=False

    def get_config(self):
        config = super().get_config()
        config.update({"n_nodes": self.n_nodes})
        return config
    
    def build(self, *args):
        #if self.include_segments:
        #    self.output_shape = [(None,), (None,)]
        #else:
        self.output_shape = [(None,),(None,)]

    def compute_output_shape(self):
        return self.output_shape

    def call(self,node_mask, edge_mask) -> list[keras.KerasTensor]:
        #TODO: write this in pure keras. The problem is that the batch_size is not known at compile-time
        # Moreover, the repeats will result in XLA compilation errors since the number of repeats is also not known at compile-time
        # Finally, since `particle_ids` will be used as segment ids, we will run into the problem that, since `num_segments` is
        # not known at compile time (since it depends on the total number of particles in a batch), the number of segments is inferred
        # from the ids. Which, in turn, means that the array returned by `segment_sum` is only as large as the largest id in `particle_ids`
        # Therefore, we need to ensure that the max possible particle id is always included.
        # There are three options to achieving this that I can think off:
        #   1) add a phantom particle that has id=max_id+1, but this will have to be cut off when particle are assigned to events
        #   2) re-order the events such that the last event is always full (but this is not always possible) and would make
        #       debugging really hard since the input order of events!=output order of predictions
        #   3) shift all ID's to ensure the last event is full (would require an additional tensor to store the mapping)  
        batch_size = tf.shape(node_mask)[0]
        n_per_event = tf.reduce_sum(tf.cast(node_mask, dtype="int32"), axis=1) #(B,)
        K = batch_size*self.n_nodes
        #find particle ids
        ids = tf.reshape(tf.range(K, dtype="int32"), (-1,self.n_nodes))
        repeats = tf.repeat(n_per_event-1, repeats=n_per_event)
        particle_ids = tf.repeat(ids[node_mask], repeats=repeats)
        neighbor_ids =  tf.repeat(tf.expand_dims(ids,axis=1), repeats=self.n_nodes, axis=1)[edge_mask]

        if self.include_segments:
            particle_seg_ids = tf.concat((particle_ids, [K-1]), axis=0) # add a phantom particle to ensure the last event is full
            #find neighbor ids
            neighbor_seg_ids = tf.concat((neighbor_ids, [K-1]), axis=0) # add a phantom particle to ensure the last event is full
            return [particle_seg_ids, neighbor_seg_ids, tf.range(K), tf.concat((tf.range(particle_ids.shape[0]), [particle_ids.shape[0]-1]), axis=0)]
        else:
            return [particle_ids, neighbor_ids]
class KNN_Features(keras.Layer):
    """
    Collects the features of the k-nearest-neighbors of each particle of each event.
    
    Parameters
    ----------
    features : tensor(N,P,C)
        features of this batch
    dists : tensor(N,P,P)
        the pair-wise distances between particles
    k: int
        K
    Returns
    -------
    tensor(N,P,K,C)
        The C features of the K nearest neighbors of the particles in the events
    """
    def __init__(self, k:int, **kwargs):
        super().__init__(**kwargs)
        self.k = k
    
    def get_config(self):
        conf = super().get_config()
        conf.update({"k": self.k})
        return conf
    
    def build(self, features_shape, dists_shape):
        self.output_shape = (None, features_shape[1], self.k, features_shape[2])

    def compute_output_shape(self):
        return self.output_shape

    def call(self, features:KerasTensor, dists:KerasTensor):
        #TODO: find a way to do this in pure keras...
        #N = ko.sum(ko.ones_like(features[:,0,0], dtype=int))
        N = tf.shape(features)[0]
        P = tf.shape(features)[1]
        _, top_k = ko.top_k(-dists,k=self.k+1, sorted=True)
        top_k = top_k[:,:,1:] #remove the node itself

        #we need to do some crazy stuff to get the right tensor containing the indices of the feature
        # vectors which we need for each particle
        #create a tensor of indices indicating the event to which each of the knn's belongs
        b_idx = ko.tile(ko.reshape(ko.arange(N, dtype=int), (N,1,1,1)), (1,P,self.k,1)) #(N,P,k,1)
        #now, create a tensor which, for each particle in each event, k pairs of indices [event_number, particle_number]
        #where particle_number corresponds to it's k-th nearest neighbor's index in the event
        idx = ko.concatenate([b_idx, ko.expand_dims(top_k, axis=3)], axis=3) #(N,P,k,2)
        return tf.gather_nd(features, idx) #now collect the features of those k nearest neighbors