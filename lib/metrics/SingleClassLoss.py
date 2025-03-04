import keras
from keras import ops as ko

class SingleClassLoss(keras.metrics.Metric):
    def __init__(self, class_label=0, name='singleclassloss', **kwargs):
        super(SingleClassLoss, self).__init__(name=name, **kwargs)
        self.class_label = class_label
        self.total_loss = self.add_weight(name=f'{name}_total_loss', initializer='zeros')
        self.count = self.add_weight(name=f'{name}_count', initializer='zeros')
        self.sample_weight_notification = False
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        if(sample_weight and not self.sample_weight_notification):
            raise NotImplementedError("Sample weights not supported by SingleClassLoss")
            self.sample_weight_notification = True #to display the notification once
        # y_true and y_pred are one-hot encoded
        # so we want the indices of rows in which the column corresponding to
        # the class label is 1
        idx = y_true[:,self.class_label]==1
        loss = keras.losses.categorical_crossentropy(y_true[idx], y_pred[idx])
        self.total_loss.assign_add(ko.sum(loss))
        self.count.assign_add(ko.cast(ko.size(loss), "float32"))
    
    def result(self):
        return self.total_loss / self.count
    
    def get_config(self):
        config = super().get_config()
        config.update({"class_label": self.class_label})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
