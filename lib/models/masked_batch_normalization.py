import tensorflow as tf

class MaskedBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        super(MaskedBatchNormalization, self).__init__(fused=False, **kwargs)

    def build(self, input_shape):
        super(MaskedBatchNormalization, self).build(input_shape) #the other one is the edge mask

    def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
        return tf.nn.weighted_moments(inputs, reduction_axes, self.edge_mask, keepdims=keep_dims)

    def call(self, inputs, edge_mask=None, **kwargs):
        if(edge_mask is None and isinstance(inputs, (tuple,list))):
            edge_mask = inputs[1]
            inputs = inputs[0]
        self.edge_mask = edge_mask #store the edge mask as an attribute to be used in _calculate_mean_and_var
        outputs = super().call(inputs,kwargs.get("training",None)) #call the normal call method.
        del self.edge_mask #don't store it in memory
        return tf.where(edge_mask,outputs,0) #apply the mask