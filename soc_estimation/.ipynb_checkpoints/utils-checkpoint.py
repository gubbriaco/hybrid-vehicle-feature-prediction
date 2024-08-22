import numpy as np
from sklearn import metrics
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt


def normalize(X):
    normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        max_value = max(X[i])
        min_value = min(X[i])
        for j in range(X.shape[1]):
            normalized_X[i][j] = ((2*(X[i][j]-min_value))/(max_value-min_value))-1
    return normalized_X


def get_metrics(y_test, y_predicted):
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