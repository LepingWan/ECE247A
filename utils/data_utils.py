import pickle
from pathlib import Path

import numpy as np


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    X = batch['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    return X.astype(float), np.array(batch['labels'])


def load_CIFAR10(ROOT):
    root = Path(ROOT)
    batches = [load_CIFAR_batch(root / ('data_batch_%d' % i)) for i in range(1, 6)]
    X_train = np.concatenate([X for X, y in batches])
    y_train = np.concatenate([y for X, y in batches])
    X_test, y_test = load_CIFAR_batch(root / 'test_batch')
    return X_train, y_train, X_test, y_test
