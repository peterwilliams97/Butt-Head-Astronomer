import cytoolz
from collections import defaultdict
import numpy as np
from keras.models import Sequential
from keras.layers import (LSTM, Dense, Embedding, Bidirectional, Dropout, GlobalMaxPool1D,
    GlobalAveragePooling1D, BatchNormalization, TimeDistributed, Flatten)
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import spacy
import os
import time
import math
from framework import MODEL_DIR, LABEL_COLS, get_n_samples_str, df_to_sentences, train_test_split
from utils import (dim, xprint, RocAucEvaluation, SaveAllEpochs, Cycler, save_model, load_model,
    save_json, load_json)
from spacy_glue import SpacySentenceWordCache


MIN, MEAN, MAX, MEAN_MAX, MEDIAN, PC75, PC90, LINEAR, LINEAR2, LINEAR3, LINEAR4, LINEAR5, EXP = (
    'MIN', 'MEAN', 'MAX', 'MEAN_MAX', 'MEDIAN', 'PC75', 'PC90', 'LINEAR', 'LINEAR2', 'LINEAR3',
    'LINEAR4', 'LINEAR5', 'EXP')
PREDICT_METHODS = (MIN, MEAN, MAX, MEAN_MAX, MEDIAN, PC75, PC90, LINEAR, LINEAR2, LINEAR3, LINEAR4,
    LINEAR5, EXP)
PREDICT_METHODS_GOOD = [MEAN, LINEAR, LINEAR2, LINEAR3]


def linear_weights(ys, limit):
    """Returns: Array of linearly increasing weights [w_1, ..., w_n]
        n = len(ys)
        sum(weights) = 1.0
        w_1 = limit
        w_n = 1.0 - limit

        w_n / w_1 = (1 / limit - 1) increases as limit decreases
    """
    n = ys.shape[0]
    weights = np.ones(n, dtype=np.float64)
    if n <= 1:
        return weights
    lo = limit
    hi = 1.0 - limit
    assert lo <= hi
    span = hi - lo
    d = span / (n - 1)
    for i in range(n):
        weights[i] = lo + d * i
    weights /= weights.sum()
    return weights


def exponential_weights(ys, limit):
    n = ys.shape[0]
    weights = np.ones(n, dtype=np.float64)
    if n <= 1:
        return weights
    lo = limit
    hi = 1.0 - limit
    assert lo <= hi
    d = hi / lo
    assert d > 1.0
    for i in range(n):
        weights[i] = lo * (d ** i)
    weights /= weights.sum()
    for i in range(n):
        assert 0.0 < weights[i] < 1.0
    return weights


def reduce(ys_in, method):
    ys = ys_in.copy()
    for j in range(ys.shape[1]):
        ys[:, j] = np.sort(ys_in[:, j])
    if method == MIN:
        return ys.min(axis=0)
    if method == MEAN:
        return ys.mean(axis=0)                 # w_n / w_1 = 1  4th BEST
    elif method == MAX:
        return ys.max(axis=0)
    elif method == MEAN_MAX:
        return (ys.mean(axis=0) + ys.max(axis=0)) / 2.0
    elif method == MEDIAN:
        return np.percentile(ys, 50.0, axis=0, interpolation='higher')
    elif method == PC75:
        return np.percentile(ys, 75.0, axis=0, interpolation='higher')
    elif method == PC90:
        return np.percentile(ys, 90.0, axis=0, interpolation='higher')
    elif method == LINEAR:
        weights = linear_weights(ys, limit=0.1)  # w_n / w_1 = 9   3rd BEST
        return np.dot(weights, ys)
    elif method == LINEAR2:
        weights = linear_weights(ys, limit=0.2)  # w_n / w_1 = 4    BEST
        return np.dot(weights, ys)
    elif method == LINEAR3:
        weights = linear_weights(ys, limit=0.3)  # w_n / w_1 = 2.3  2nd BEST
        return np.dot(weights, ys)
    elif method == LINEAR4:
        weights = linear_weights(ys, limit=0.05)  # w_n / w_1 = 19
        return np.dot(weights, ys)
    elif method == LINEAR5:
        weights = linear_weights(ys, limit=0.01)  # w_n / w_1 = 99
        return np.dot(weights, ys)
    elif method == EXP:
        weights = exponential_weights(ys, limit=0.3)
        y = np.dot(weights, ys)
        assert len(y.shape) == 1 and len(y) == ys.shape[1], (weights.shape, ys.shape, y.shape)
        return y
    raise ValueError('Bad method=%s' % method)


if False:

    def test_reduce(ys):
        print('-' * 80)
        print(ys)
        for method in PREDICT_METHODS:
            y = reduce(ys, method)
            print('%8s: %s' % (method, y))

    ys = np.ones((1, 4), dtype=np.float32)
    test_reduce(ys)

    ys = np.ones((6, 4), dtype=np.float32)
    ys[:, 0] = 0.0
    ys[:3, 2] = -1.0
    ys[:3, 1] = 2.0
    test_reduce(ys)
    assert False
