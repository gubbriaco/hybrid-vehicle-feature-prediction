from tensorflow import keras
from tensorflow.keras import layers


class CustomLeakyReLU(layers.Layer):
    """
    Custom implementation of the Leaky ReLU activation function.

    This layer applies the Leaky ReLU activation function to its inputs, allowing a small gradient when the input is negative.
    
    Args:
        negative_slope (float): The slope of the activation function for negative inputs. Must be a non-negative float.
        **kwargs: Additional keyword arguments to be passed to the parent class.
    
    Attributes:
        negative_slope (float): The slope of the activation function for negative inputs.
        leaky_relu (tf.keras.layers.LeakyReLU): The Keras LeakyReLU activation function layer.
    
    Methods:
        build(input_shape): Initializes the layer. Required for Keras layers, but this implementation does not add any weights.
        call(inputs): Applies the Leaky ReLU activation function to the inputs.
        get_config(): Returns the configuration of the layer, including the `negative_slope` parameter.
        from_config(config): Creates an instance of `CustomLeakyReLU` from the given configuration dictionary.
    """
    
    def __init__(self, negative_slope, **kwargs):
        super(CustomLeakyReLU, self).__init__(**kwargs)
        self.negative_slope = negative_slope
        self.leaky_relu = layers.LeakyReLU(negative_slope=negative_slope)

    def build(self, input_shape):
        super(CustomLeakyReLU, self).build(input_shape)

    def call(self, inputs):
        """
        Applies the Leaky ReLU activation function to the inputs.

        Args:
            inputs (tf.Tensor): The input tensor to which the activation function is applied.

        Returns:
            tf.Tensor: The output tensor after applying the Leaky ReLU function.
        """
        return self.leaky_relu(inputs)

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: A dictionary containing the configuration of the layer, including the `negative_slope` parameter.
        """
        config = super(CustomLeakyReLU, self).get_config()
        config.update({"negative_slope": self.negative_slope})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates an instance of `CustomLeakyReLU` from the given configuration dictionary.

        Args:
            config (dict): The configuration dictionary containing the parameters to initialize the layer.

        Returns:
            CustomLeakyReLU: An instance of `CustomLeakyReLU` initialized with the given configuration.
        """
        return cls(**config)


class CustomClippedReLU(layers.Layer):
    """
    Custom implementation of the Clipped ReLU activation function.

    This layer applies the Clipped ReLU activation function to its inputs, clipping the output values to be within the range [0, 1].
    
    Args:
        **kwargs: Additional keyword arguments to be passed to the parent class.
    
    Methods:
        build(input_shape): Initializes the layer. Required for Keras layers, but this implementation does not add any weights.
        call(inputs): Applies the Clipped ReLU activation function to the inputs.
        get_config(): Returns the configuration of the layer (empty in this implementation).
        from_config(config): Creates an instance of `CustomClippedReLU` from the given configuration dictionary.
    """
    
    def __init__(self, **kwargs):
        super(CustomClippedReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CustomClippedReLU, self).build(input_shape)

    def call(self, inputs):
        """
        Applies the Clipped ReLU activation function to the inputs.

        Args:
            inputs (tf.Tensor): The input tensor to which the activation function is applied.

        Returns:
            tf.Tensor: The output tensor after applying the Clipped ReLU function, clipped to the range [0, 1].
        """
        return keras.backend.clip(inputs, 0, 1)

    def get_config(self):
        """
        Returns the configuration of the layer.

        Returns:
            dict: An empty dictionary, as this layer does not have configurable parameters.
        """
        config = super(CustomClippedReLU, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates an instance of `CustomClippedReLU` from the given configuration dictionary.

        Args:
            config (dict): The configuration dictionary containing the parameters to initialize the layer.

        Returns:
            CustomClippedReLU: An instance of `CustomClippedReLU` initialized with the given configuration.
        """
        return cls(**config)
