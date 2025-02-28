"""
Simple callback to save model at regular intervals.
"""
import keras
import os

class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, freq, directory, name_fmt:str, save_weights:bool=False):
        super().__init__()
        self.freq = freq
        self.directory = directory
        self.name_fmt = name_fmt
        self.save_weights = save_weights

    def on_epoch_begin(self, epoch, logs=None):
        if self.freq > 0 and epoch>0 and epoch % self.freq == 0:
            file = os.path.join(self.directory, self.name_fmt.format(epoch=epoch)+".keras")
            print(f"SaveModelCallback epoch {epoch:d}: Saving model to {file:s}")
            self.model.save(file)
            if self.save_weights:
                self.model.save_weights(file.replace(".keras", ".weights.h5"))