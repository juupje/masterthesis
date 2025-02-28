"""
Implements custom models for weakly supervised learning.
The only difference to standard keras Models is that they have additional metrics for the different classes.
"""
import keras
from metrics.SingleClassLoss import SingleClassLoss

class WeakModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sb_loss = SingleClassLoss(class_label=0, name='background_region_loss')
        self.sr_loss = SingleClassLoss(class_label=1, name='signal_region_loss')
        self.bg_loss = SingleClassLoss(class_label=0, name='bg_loss')
        self.sn_loss = SingleClassLoss(class_label=1, name='sn_loss')
        self.extra_metrics = [self.sb_loss, self.sr_loss, self.bg_loss, self.sn_loss]
    
    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        if(isinstance(y, (tuple, list))):
            y_weak, y_true = y
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss = self.compute_loss(y=y_weak, y_pred=y_pred, sample_weight=sample_weight)
        # Update the metrics.
        for metric in self.metrics:
            if(metric.name == "loss"): metric.update_state(loss)
            else: metric.update_state(y_true=y_weak, y_pred=y_pred, sample_weight=sample_weight)
        self.sb_loss.update_state(y_weak, y_pred)
        self.sr_loss.update_state(y_weak, y_pred, sample_weight=sample_weight)
        self.bg_loss.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.sn_loss.update_state(y_true, y_pred, sample_weight=sample_weight)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        results = {m.name: m.result() for m in self.metrics}
        results.update({m.name: m.result() for m in self.extra_metrics})
        return results
    
    def get_config(self):
        return super().get_config()

class SupervisedModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bg_loss = SingleClassLoss(class_label=0, name='bg_loss')
        self.sn_loss = SingleClassLoss(class_label=1, name='sn_loss')
        self.extra_metrics = [self.bg_loss, self.sn_loss]

    def test_step(self, data):
        if len(data) == 3:
            x, y_true, sample_weight = data
        else:
            sample_weight = None
            x, y_true = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        loss = self.compute_loss(y=y_true, y_pred=y_pred, sample_weight=sample_weight)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name == "loss": metric.update_state(loss)
            else: metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.bg_loss.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.sn_loss.update_state(y_true, y_pred, sample_weight=sample_weight)
        #self.roc_metric.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        results = {m.name: m.result() for m in self.metrics}
        results.update({m.name: m.result() for m in self.extra_metrics})
        return results
    
    def get_config(self):
        return super().get_config()