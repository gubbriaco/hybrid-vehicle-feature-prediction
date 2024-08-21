import numpy as np


def normalize(X):
    normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        max_value = max(X[i])
        min_value = min(X[i])
        for j in range(X.shape[1]):
            normalized_X[i][j] = ((2*(X[i][j]-min_value))/(max_value-min_value))-1
    return normalized_X


def rmse(observed, predicted):
    diff = []
    if len(predicted) != len(observed):
        raise Exception(f'(len(predicted),len(observed))=({len(predicted)},{len(observed)}) => {len(predicted)}!={len(observed)}')
    for i in range(len(predicted)):
        diff_value = predicted[i] - observed[i]
        quad_value = diff_value**2
        diff.append(quad_value)
    return np.sqrt(np.mean(diff))


def maxv(observed, predicted):
    diff = []
    if len(predicted) != len(observed):
        raise Exception(f'(len(predicted),len(observed))=({len(predicted)},{len(observed)}) => {len(predicted)}!={len(observed)}')
    for i in range(len(predicted)):
        diff_value = predicted[i] - observed[i]
        diff.append(diff_value)
    return np.max(np.abs(diff))


def mae(observed, predicted):
    diff = []
    if len(predicted) != len(observed):
        raise Exception(f'(len(predicted),len(observed))=({len(predicted)},{len(observed)}) => {len(predicted)}!={len(observed)}')
    for i in range(len(predicted)):
        diff_value = predicted[i] - observed[i]
        diff.append(diff_value)
    return np.mean(np.abs(diff))
