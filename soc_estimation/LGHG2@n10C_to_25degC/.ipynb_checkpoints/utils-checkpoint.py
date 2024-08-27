import numpy as np
from sklearn import metrics
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt


def normalize(X):
    """
    Normalize the input data matrix `X` using min-max scaling to a range of [-1, 1].

    This function performs normalization by scaling each feature of the data matrix 
    `X` to a range between -1 and 1. The normalization formula used is:
    ((2*(X[i][j] - min_value)) / (max_value - min_value)) - 1, where `min_value` 
    and `max_value` are the minimum and maximum values of the feature in each row.

    Parameters:
    X (numpy.ndarray): A 2D numpy array where rows represent samples and columns represent features.

    Returns:
    numpy.ndarray: A 2D numpy array of the same shape as `X` with normalized values.
    """
    normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        max_value = max(X[i])
        min_value = min(X[i])
        for j in range(X.shape[1]):
            normalized_X[i][j] = ((2*(X[i][j]-min_value))/(max_value-min_value))-1
    return normalized_X


def get_metrics(y_test, y_predicted):
    """
    Calculate and format various metrics between true and predicted values.

    This function computes a set of standard regression metrics including max error, mean 
    absolute error, mean absolute percentage error, mean squared error, root mean squared error, 
    and root mean squared logarithmic error. The results are returned as a styled DataFrame.

    Parameters:
    y_test (array-like): True values of the target variable.
    y_predicted (array-like): Predicted values of the target variable.

    Returns:
    pandas.io.formats.style.Styler: A styled DataFrame containing the metrics and their values.
    """
    results = {
        "Metric": [
            "max_error",
            "mean_absolute_error",
            "mean_absolute_percentage_error",
            "mean_squared_error",
            "root_mean_squared_error",
            "root_mean_squared_log_error"
        ],
        "Value": [
            metrics.max_error(y_test, y_predicted),
            metrics.mean_absolute_error(y_test, y_predicted),
            metrics.mean_absolute_percentage_error(y_test, y_predicted),
            metrics.mean_squared_error(y_test, y_predicted),
            metrics.root_mean_squared_error(y_test, y_predicted),
            metrics.root_mean_squared_log_error(y_test, y_predicted)
        ]
    }
    
    df = pd.DataFrame(results)
    
    styled_table = df.style.background_gradient(cmap="coolwarm").set_properties(**{
        'border': '1.5px solid black', 
        'color': 'black',
        'font-size': '12pt',
        'text-align': 'center'
    })
    
    return styled_table


def plot_obs_pred(y_test, y_predicted, ylabel, xlabel):
    """
    Plot observed vs predicted values.

    This function generates two subplots. The first subplot displays the full range of observed and 
    predicted values, while the second subplot zooms in on the range [0, 10000] on the x-axis 
    and [0, 1] on the y-axis. Both subplots include legends to differentiate between observed 
    and predicted values.

    Parameters:
    y_test (array-like): True values of the target variable.
    y_predicted (array-like): Predicted values of the target variable.
    ylabel (str): Label for the y-axis.
    xlabel (str): Label for the x-axis.

    Returns:
    None: The function displays the plots directly.
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11, 6))

    axs[0].plot(y_test, label='observed')
    axs[0].plot(y_predicted, label='predicted')
    axs[0].set_ylabel(f'{ylabel}')
    axs[0].set_xlabel(f'{xlabel}')
    axs[0].legend()
    
    axs[1].plot(y_test, label='observed')
    axs[1].plot(y_predicted, label='predicted')
    axs[1].set_ylabel(f'{ylabel}')
    axs[1].set_xlabel(f'{xlabel}')
    axs[1].legend()
    axs[1].axis([0, 10000, 0, 1])
    
    plt.tight_layout()
    plt.show()
