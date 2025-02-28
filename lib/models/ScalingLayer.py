import keras

class ScalingLayer(keras.layers.Layer):
    """
    Not sure where this is used, but I don't want to remove it
    """
    def __init__(self, scaling=1., **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        if(scaling is None):
            self.scale = None
        if(isinstance(scaling, str)):
            if(scaling=="zero"):
                self.scale = keras.Variable(0.)
            elif(scaling=="one" or scaling=="identity"):
                self.scale = keras.Variable(1.)
            else:
                raise ValueError(f"Unknown scaling {scaling}")
        else:
            self.scale = keras.Variable(scaling)
        self.scaling = scaling

    def call(self, inputs):
        if(self.scale is not None):
            return inputs * self.scale
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"scaling": self.scaling})
        return config
