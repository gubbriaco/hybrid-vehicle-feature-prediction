"""
This module provides classes for building Feedforward Neural Network (FNN) models using Keras,
custom activation functions, and an Adaptive H-Infinity Filter (AHIF) for estimating the state of a system
based on noisy measurements.

Main Classes:
- FNN: Implements a Feedforward Neural Network model with Keras.
- CustomLeakyReLU: A custom implementation of the Leaky ReLU activation function.
- CustomClippedReLU: A custom implementation of the Clipped ReLU activation function.
- AHIF: An Adaptive H-Infinity Filter for state estimation with noise.
"""


import sys
import logging
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np



class FNN:
    """
    A class to represent a Feedforward Neural Network (FNN) model.
    
    Attributes:
        input_shape (int): The number of features in the input data.
        output_shape (int): The number of output neurons (e.g., number of classes for classification).
        model (keras.Model, optional): The Keras model instance. Defaults to None.
    """
    
    def __init__(self, input_shape, output_shape):
        """
        Initializes the FNNModel with the given input and output shapes.
        
        Args:
            input_shape (int): The number of features in the input data.
            output_shape (int): The number of output neurons.
        """
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.input_shape = input_shape
        self.output_shape = output_shape
        logging.info("FNNModel instance created with input shape %d and output shape %d.", input_shape, output_shape)
        self.model = None

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.output_shape
    
    def get_model(self):
        """
        Returns the trained model instance.
        
        Returns:
            keras.Model: The trained Keras model instance.
        """
        return self.model
        
    def build(self):
        """
        Builds the model architecture. This method constructs the layers and specifies the activation functions.
        """
        logging.info("Building the model...")
        self.model = keras.Sequential([
            layers.Input(
                shape=(self.input_shape,)
            ),
            layers.Dense(
                256,
                activation=keras.activations.relu
            ),
            layers.Dense(
                256,
                activation=keras.activations.relu
            ),
            layers.Dense(
                128,
                activation=CustomLeakyReLU(negative_slope=0.3)
            ),
            layers.Dense(
                self.output_shape,
                activation=CustomClippedReLU()
            )
        ])
        logging.info("Model built successfully.")

    def compile(self):
        """
        Compiles the model with the specified optimizer and loss function. 
        This method must be called after `build_model` to prepare the model for training.
        
        Raises:
            ValueError: If the model is not built before calling this method.
        """
        logging.info("Compiling the model...")
        if self.model is None:
            raise ValueError("Model is not built. Call `build_model()` before `compile_model()`.")
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=False
        )
        optimizer = keras.optimizers.SGD(
            learning_rate=lr_schedule
        )
        self.model.compile(
            optimizer=optimizer,
            loss='mse'
        )
        logging.info("Model compiled successfully.")
        self.model.summary()

    
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


class AHIF:
    """
    Adaptive H-Infinity Filter (AHIF) Implementation.

    This class implements a simplified Adaptive H-Infinity Filter for estimating the state of a system
    based on noisy measurements. The process variance is adapted based on the residuals over time.

    Attributes:
        process_variance (float): The variance of the process (model uncertainty).
        measurement_variance (float): The variance of the measurement (sensor noise).
        estimate (float): The current estimate of the state.
        error_covariance (float): The current error covariance associated with the estimate.
    """
    def __init__(self, process_variance=1e-5, measurement_variance=1e-1, initial_estimate=0, initial_error_covariance=1):
        """
        Initializes the Adaptive H-Infinity Filter with given parameters.

        Args:
            process_variance (float): Initial variance of the process (default is 1e-5).
            measurement_variance (float): Variance of the measurement noise (default is 1e-1).
            initial_estimate (float): Initial estimate of the system state (default is 0).
            initial_error_covariance (float): Initial error covariance (default is 1).
        """
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self._process_variance = process_variance
        self._measurement_variance = measurement_variance
        self._estimate = initial_estimate
        self._error_covariance = initial_error_covariance
        logging.info(f"Initialized AHIF with process_variance={process_variance}, measurement_variance={measurement_variance}, "
                      f"initial_estimate={initial_estimate}, initial_error_covariance={initial_error_covariance}")

    def _update(self, measurement):
        """
        Performs the prediction and update steps of the filter for a single measurement.

        Args:
            measurement (float): The latest measurement from the system.

        Returns:
            float: The updated estimate of the system state.
        """
        # Prediction
        predicted_estimate = self._estimate
        predicted_error_covariance = self._error_covariance + self._process_variance
        kalman_gain = predicted_error_covariance / (predicted_error_covariance + self._measurement_variance)
        # Update estimate and error covariance
        self._estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self._error_covariance = (1 - kalman_gain) * predicted_error_covariance
        return self._estimate

    def _adapt(self, residuals):
        """
        Adapts the process variance based on the recent residuals (measurement - estimate).

        Args:
            residuals (list of float): The list of residuals from recent measurements.
        """
        if len(residuals) > 1:
            residual_std = np.std(residuals)
            # Prevent process variance from becoming too small
            self._process_variance = max(residual_std ** 2, 1e-5)
        else:
            # Default behavior if residuals are not enough for adaptation
            self._process_variance = 1e-5

    def apply(self, data):
        """
        Applies the Adaptive H-Infinity Filter to a series of measurements.

        Args:
            data (list of float): The list of measurements.

        Returns:
            numpy.ndarray: The list of state estimates corresponding to the measurements.
        """
        estimates = []
        residuals = []
        self._estimate = data[0]
        for measurement in data:
            estimate = self._update(measurement)
            estimates.append(estimate)
            residual = measurement - estimate
            residuals.append(residual)
            # Adapt filter based on the latest residuals
            if len(residuals) > 10:
                self._adapt(residuals[-10:])
        return np.array(estimates)
