# Data loading utilities
import numpy as np
import os
import csv

def train_test_split(X, y, test_ratio=0.3, seed=42):
    np.random.seed(seed)

    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio);

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
