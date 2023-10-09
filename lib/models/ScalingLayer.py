import tensorflow as tf

class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, scaling=1., **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        if(scaling is None):
            self.scale = None
        if(isinstance(scaling, str)):
            if(scaling=="zero"):
                self.scale = tf.Variable(0.)
            elif(scaling=="one" or scaling=="identity"):
                self.scale = tf.Variable(1.)
            else:
                raise ValueError(f"Unknown scaling {scaling}")
        else:
            self.scale = tf.Variable(scaling)
        self.scaling = scaling

    def call(self, inputs):
        if(self.scale is not None):
            return inputs * self.scale
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"scaling": self.scaling})
        return config
