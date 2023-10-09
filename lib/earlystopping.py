import tensorflow as tf
from keras.utils import io_utils
import numpy as np

class EarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, min_epoch, max_fluct:float=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.min_epoch = min_epoch
        self.stopped = False
        self.max_fluct = max_fluct
        self.prev = None

    def on_epoch_end(self, epoch, logs=None):
        if(epoch < self.min_epoch): return
        if(self.prev is None): self.prev = self.best
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0
        
        if tf.math.abs(current-self.prev) > self.max_fluct:
            # too much fluctuation! Restart
            self.wait = 0
            self.best = current
            self.best_epoch = epoch

        # Only check after the first min_epochs.
        if self.wait >= self.patience and epoch > self.min_epoch:
            self.stopped_epoch = epoch
            self.stopped = True
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    io_utils.print_msg(
                        "Restoring model weights from "
                        "the end of the best epoch: "
                        f"{self.best_epoch + 1}."
                    )
                self.model.set_weights(self.best_weights)
        self.prev = current