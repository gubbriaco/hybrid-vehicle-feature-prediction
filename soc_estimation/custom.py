from tensorflow import keras
from tensorflow.keras import layers


class CustomLeakyReLU(layers.Layer):
    def __init__(self, negative_slope, **kwargs):
        super(CustomLeakyReLU, self).__init__(**kwargs)
        self.negative_slope = negative_slope
        self.leaky_relu = layers.LeakyReLU(negative_slope=negative_slope)

    def build(self, input_shape):
        super(CustomLeakyReLU, self).build(input_shape)

    def call(self, inputs):
        return self.leaky_relu(inputs)

    def get_config(self):
        config = super(CustomLeakyReLU, self).get_config()
        config.update({"negative_slope": self.negative_slope})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomClippedReLU(layers.Layer):
    def __init__(self, **kwargs):
        super(CustomClippedReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CustomClippedReLU, self).build(input_shape)

    def call(self, inputs):
        return keras.backend.clip(inputs, 0, 1)

    def get_config(self):
        config = super(CustomClippedReLU, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)