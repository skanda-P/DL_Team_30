

from itertools import combinations
import numpy as np
from perceptron import Perceptron

class OneVsOneClassifier:

    def __init__(self, activation="logistic", learning_rate=0.01, epochs=1000):
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.classifiers = {}
        self.classes_ = None

    def _decision_threshold(self, model):
        return 0.5 if model.activation_name == "logistic" else 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.classifiers = {}

        for class_a, class_b in combinations(self.classes_, 2):
            mask = (y == class_a) | (y == class_b)
            X_pair = X[mask]
            y_pair_raw = y[mask]


            y_pair = np.where(y_pair_raw == class_a, 0.0, 1.0)

            model = Perceptron(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                activation=self.activation,
            )
            model.fit(X_pair, y_pair)

            self.classifiers[(class_a, class_b)] = model

        return self

    def predict_pair_labels(self, class_a, class_b, X):
        X = np.asarray(X, dtype=float)
        model = self.classifiers[(class_a, class_b)]
        threshold = self._decision_threshold(model)
        raw = model.predict(X)
        hard = (raw >= threshold).astype(int)
        return np.where(hard == 0, class_a, class_b)

    def predict(self, X):
        if self.classes_ is None:
            raise ValueError("Model has not been trained yet.")

        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_index = {c: i for i, c in enumerate(self.classes_)}

        votes = np.zeros((n_samples, n_classes), dtype=int)

        for (class_a, class_b), model in self.classifiers.items():
            threshold = self._decision_threshold(model)
            raw = model.predict(X)
            hard = (raw >= threshold).astype(int)

            votes[hard == 0, class_index[class_a]] += 1
            votes[hard == 1, class_index[class_b]] += 1

        winners = np.argmax(votes, axis=1)
        return self.classes_[winners]
