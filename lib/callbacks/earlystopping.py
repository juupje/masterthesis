"""
author: Joep Geuskens
"""
import keras
import numpy as np

class EarlyStopping(keras.callbacks.EarlyStopping):
    """
    Early stopping with a minimum number of epochs before stopping.
    """
    def __init__(self, min_epoch, max_fluct:float=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.min_epoch = min_epoch
        self.stopped = False
        self.max_fluct = max_fluct
        self.prev = None

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op is None:
            self._set_monitor_op()

        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch: return
        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0
        """
        if self.prev and keras.ops.abs(current-self.prev) > self.max_fluct:
            # too much fluctuation! Restart
            self.wait = 0
            self.best = current
            self.best_epoch = epoch
        """

        # Only check after the first min_epochs.
        if self.wait >= self.patience and epoch > self.min_epoch:
            self.stopped_epoch = epoch
            self.stopped = True
            self.model.stop_training = True
        self.prev = current
