
import numpy as np
import os
import csv

def train_test_split(X, y, test_ratio=0.3, seed=42, rng=None):
    """
    Splits (X, y) into train/test sets using a random permutation.

    If `rng` is provided (a np.random.Generator), it is used directly and its
    state advances with the call - this lets callers (e.g.
    stratified_train_test_split) share one generator across multiple calls so
    each call draws a genuinely different permutation instead of restarting
    from the same seed. If `rng` is not provided, a fresh local generator is
    created from `seed`. Either way, global numpy random state is never
    touched, so this function can't affect unrelated random code elsewhere
    in the pipeline.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))
    test_size = int(len(X) * test_ratio)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def load_LS_data(par_dir):

    X = []
    y = []
    class_files = ['Class1.txt', 'Class2.txt', 'Class3.txt']

    for label, class_file in enumerate(class_files):
        file_path = os.path.join(par_dir, class_file)
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    x1, x2 = map(float, parts)
                    X.append([x1, x2])
                    y.append(label)

    return np.array(X), np.array(y)

def load_nls_data(filepath):
    X = []
    y = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = [line.strip() for line in lines[1:] if line.strip()]
    for index, line in enumerate(data_lines):
        parts = line.split()
        if len(parts) >= 2:
            X.append([float(parts[0]), float(parts[1])])
            if index < 500:
                y.append(0)
            elif index < 1000:
                y.append(1)
            else:
                y.append(2)

    return np.array(X), np.array(y)


def load_regression_csv(filepath):
    X = []
    y = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_idx = 0
    if lines:
        try:
            float(lines[0].strip().split(',')[0])
        except ValueError:
            start_idx = 1

    for line in lines[start_idx:]:
        line = line.strip()
        if line:
            parts = [float(val) for val in line.split(',')]
            y.append(parts[-1])
            X.append(parts[:-1])

    return np.array(X), np.array(y)


def stratified_train_test_split(X, y, test_ratio=0.3, seed=42):
    rng = np.random.default_rng(seed)

    X_train_parts, X_test_parts = [], []
    y_train_parts, y_test_parts = [], []

    for cls in np.unique(y):
        mask = y == cls
        X_cls, y_cls = X[mask], y[mask]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_cls, y_cls, test_ratio=test_ratio, rng=rng
        )

        X_train_parts.append(X_tr)
        X_test_parts.append(X_te)
        y_train_parts.append(y_tr)
        y_test_parts.append(y_te)

    X_train = np.vstack(X_train_parts)
    X_test = np.vstack(X_test_parts)
    y_train = np.concatenate(y_train_parts)
    y_test = np.concatenate(y_test_parts)

    return X_train, X_test, y_train, y_test
