from tensorflow import keras
import os

class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, freq, directory, name_fmt):
        super().__init__()
        self.freq = freq
        self.directory = directory
        self.name_fmt = name_fmt
    def on_epoch_begin(self, epoch, logs=None):
        if self.freq > 0 and epoch>0 and epoch % self.freq == 0:
            file = os.path.join(self.directory, self.name_fmt.format(epoch=epoch))
            print(f"SaveModelCallback epoch {epoch:d}: Saving model to {file:s}")
            self.model.save(file)