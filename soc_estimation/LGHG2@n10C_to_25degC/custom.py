from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


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
    Adaptive Hybrid Iterative Filter (AHIF) for data processing using adaptive Kalman filtering.

    This class implements a Kalman filter that dynamically adapts to changes in measurement residual variability.
    The filter updates its estimate and process variance based on the provided data, improving estimate accuracy over time.

    Attributes:
        _process_variance (float): Initial process variance (default: 1e-5).
        _measurement_variance (float): Measurement variance (default: 1e-1).
        _estimate (float): Initial estimate value (default: 0).
        _error_covariance (float): Initial error covariance (default: 1).
    """

    def __init__(self, process_variance=1e-5, measurement_variance=1e-1, initial_estimate=0, initial_error_covariance=1):
        """
        Initializes an AHIF instance.

        Args:
            process_variance (float): Variance of the process affecting the filter's prediction (default: 1e-5).
            measurement_variance (float): Variance of the measurement affecting the filter's update (default: 1e-1).
            initial_estimate (float): Initial value of the estimate (default: 0).
            initial_error_covariance (float): Initial error covariance (default: 1).
        """
        self._process_variance = process_variance
        self._measurement_variance = measurement_variance
        self._estimate = initial_estimate
        self._error_covariance = initial_error_covariance

    def _update(self, measurement):
        """
        Updates the estimate and error covariance based on the provided measurement.

        This method predicts the new state and updates the estimate and error covariance using the Kalman gain.

        Args:
            measurement (float): The current measurement to update the estimate with.

        Returns:
            float: The updated estimate.
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
        Adapts the process variance based on recent residuals.

        This method updates the process variance (process_variance) based on the standard deviation of the residuals.
        The process variance is adjusted to avoid becoming too small, which could destabilize the filter.

        Args:
            residuals (list of float): List of recent residuals used for adaptation.
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
        Applies the AHIF to the provided data and returns the resulting estimates.

        This method performs the estimate update for each measurement in the data and adapts the filter based on recent residuals.

        Args:
            data (list of float): List of measurement data to which the filter will be applied.

        Returns:
            numpy.ndarray: A numpy array containing the resulting estimates for each data point.
        """
        estimates = []
        residuals = []
        for measurement in data:
            estimate = self._update(measurement)
            estimates.append(estimate)
            residual = measurement - estimate
            residuals.append(residual)
            # Adapt filter based on the latest residuals
            if len(residuals) > 10:
                self._adapt(residuals[-10:])
        return np.array(estimates)
