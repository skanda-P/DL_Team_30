import numpy as np
import os
import csv


def train_test_split(X, y, test_ratio=0.3, seed=42, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))
    test_size = int(len(X) * test_ratio)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


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


def stratified_three_way_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    # Splits data into train, val, and test per class using a shared rng
    rng = np.random.default_rng(seed)

    X_train_parts, X_val_parts, X_test_parts = [], [], []
    y_train_parts, y_val_parts, y_test_parts = [], [], []

    test_ratio = 1.0 - train_ratio - val_ratio
    val_ratio_of_rem = val_ratio / (train_ratio + val_ratio)

    for cls in np.unique(y):
        mask = y == cls
        X_cls, y_cls = X[mask], y[mask]

        X_rem, X_te, y_rem, y_te = train_test_split(
            X_cls, y_cls, test_ratio=test_ratio, rng=rng
        )

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_rem, y_rem, test_ratio=val_ratio_of_rem, rng=rng
        )

        X_train_parts.append(X_tr)
        X_val_parts.append(X_va)
        X_test_parts.append(X_te)
        y_train_parts.append(y_tr)
        y_val_parts.append(y_va)
        y_test_parts.append(y_te)

    X_train = np.vstack(X_train_parts)
    X_val = np.vstack(X_val_parts)
    X_test = np.vstack(X_test_parts)
    y_train = np.concatenate(y_train_parts)
    y_val = np.concatenate(y_val_parts)
    y_test = np.concatenate(y_test_parts)

    return X_train, X_val, X_test, y_train, y_val, y_test


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
