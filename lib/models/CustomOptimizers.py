import keras
from keras import ops as ko

"""NOTE: This functionality is available in keras v3 and should be used instead"""
class BatchedAdamW(keras.optimizers.AdamW):
    def __init__(self, update_interval:int=1, **kwargs):
        raise DeprecationWarning("""BatchedAdamW is depricated, use the `gradient_accumulation_steps`
                                parameter of the keras built-in AdamW optimizer""")
        super().__init__(**kwargs)
        assert update_interval>=1 and type(update_interval)==int
        self.update_interval = update_interval
        self._accumulated_gradients = []
        self._is_built = False
        with keras.name_scope(self.name):
            self.update_counter = keras.Variable(0, name="update_count", dtype="int16", trainable=False)

    def _reset_accumulated_gradients(self):
        for g in self._accumulated_gradients:
            g.assign(ko.zeros_like(g))
        self.update_counter.assign(0)

    def build(self, var_list):
        super().build(var_list)
        if(hasattr(self, "_is_built") and self._is_built): return
        if(self.update_interval > 1):
            self._accumulated_gradients = [
                self.add_variable_from_reference(model_variable=var, variable_name="gradient_accum")
                for var in var_list]
        self._is_built = True

    def apply_gradients(self, grads_and_vars, name=None, skip_gradients_aggregation=False, **kwargs):
        #this is copied from the optimizer class!
        if not skip_gradients_aggregation:
            grads_and_vars = self.aggregate_gradients(grads_and_vars) #aggregation over devices! not batches!
        if(self.update_interval==1 or not(hasattr(self, "_is_built") and self._is_built)):
            return super().apply_gradients(grads_and_vars, name, skip_gradients_aggregation=False, **kwargs)
        self.update_counter.assign_add(1)
        _grads, _vars = list(zip(*grads_and_vars))

        def _accumulate_gradients(self, grads):
            for i in range(len(grads)):
                self._accumulated_gradients[i].assign_add(grads[i])
            return self.iterations

        def _update_fn(self, _grads, _vars):
            #tf.print(f"Updating optimizer, {self.iterations}")
            grads = [(_grads[i] + self._accumulated_gradients[i])/self.update_interval for i in range(len(_grads))]
            grads_and_vars = zip(grads, _vars)
            res = super().apply_gradients(grads_and_vars, name, skip_gradients_aggregation=False, **kwargs)
            self._reset_accumulated_gradients()
            return res
        return ko.cond(self.update_counter % self.update_interval==0, true_fn = lambda: _update_fn(self, _grads, _vars), false_fn=lambda: _accumulate_gradients(self, _grads))

    def get_config(self):
        config = super().get_config()
        config.update({"update_interval": self.update_interval})
        return config