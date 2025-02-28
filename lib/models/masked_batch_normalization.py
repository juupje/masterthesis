import keras
from keras import ops as ko

# Copied from https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/nn_impl.py#L1320
def weighted_moments(x, axes, frequency_weights, name=None, keepdims=None):
  """Returns the frequency-weighted mean and variance of `x`.

  Args:
    x: A tensor.
    axes: 1-d tensor of int32 values; these are the axes along which
      to compute mean and variance.
    frequency_weights: A tensor of positive weights which can be
      broadcast with x.
    name: Name used to scope the operation.
    keepdims: Alias of keep_dims.

  Returns:
    Two tensors: `weighted_mean` and `weighted_variance`.
  """
  if keep_dims is None:
    keep_dims = False
  with keras.name_scope(name, "weighted_moments", [x, frequency_weights, axes]):
    x = ko.convert_to_tensor(x, name="x")
    frequency_weights = ko.convert_to_tensor(frequency_weights, name="frequency_weights")

    # Unlike moments(), this just uses a simpler two-pass method.

    # See comment in moments() WRT precision; it applies here too.
    needs_cast = keras.backend.standardize_dtype(x.dtype) == 'float16'
    if needs_cast: x = ko.cast(x, 'float32')

    if frequency_weights.dtype != x.dtype:
      frequency_weights = ko.cast(frequency_weights, x.dtype)

    # Note that we use keep_dims=True for our reductions regardless of the arg;
    # this is so that the results remain broadcast-compatible with the inputs.
    weighted_input_sum = ko.sum(frequency_weights * x, axes, name="weighted_input_sum", keepdims=True)

    # The shape of the weights isn't necessarily the same as x's
    # shape, just broadcast-compatible with it -- so this expression
    # performs broadcasting to give a per-item weight, with the same
    # shape as (frequency_weights * x). This avoids having to reason
    # through all the broadcast logic to compute a correct
    # sum_of_weights.
    broadcasted_weights = frequency_weights + ko.zeros_like(x)

    sum_of_weights = ko.sum(broadcasted_weights, axes, name="sum_of_weights", keepdims=True)

    weighted_mean = ko.divide_no_nan(weighted_input_sum, sum_of_weights)

    # Have the weighted mean; now on to variance:
    weighted_distsq = ko.sum(frequency_weights * ko.square(x - weighted_mean),
                                axes, name="weighted_distsq", keepdims=True)

    weighted_variance = ko.divide_no_nan(weighted_distsq, sum_of_weights)

    if not keep_dims:
      weighted_mean = ko.squeeze(weighted_mean, axis=axes)
      weighted_variance = ko.squeeze(
          weighted_variance, axis=axes)

    if needs_cast:
      weighted_mean = ko.cast(weighted_mean, 'float16')
      weighted_variance = ko.cast(weighted_variance, 'float16')
    return weighted_mean, weighted_variance

class MaskedBatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        super(MaskedBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskedBatchNormalization, self).build(input_shape) #the other one is the edge mask

    #Overridden from BatchNorm, called by super.call
    def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
        return weighted_moments(inputs, reduction_axes, self.edge_mask, keepdims=keep_dims)

    def call(self, inputs, edge_mask=None, **kwargs):
        if(edge_mask is None and isinstance(inputs, (tuple,list))):
            edge_mask = inputs[1]
            inputs = inputs[0]
        self.edge_mask = edge_mask #store the edge mask as an attribute to be used in _calculate_mean_and_var
        outputs = super().call(inputs,kwargs.get("training",None)) #call the normal call method.
        return ko.where(edge_mask,outputs,0) #apply the mask