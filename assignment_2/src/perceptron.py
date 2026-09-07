

"""
Perceptron implementation for Deep Learning Assignment 1.

Supports:
    - Logistic activation for classification
    - Tanh activation for classification
    - Linear activation for regression

Training is performed using gradient descent.
"""

import numpy as np

from utils.activations import ACTIVATIONS


class Perceptron:
    """
    Single-layer perceptron trained using gradient descent.
    """

    def __init__(
        self,
        learning_rate=0.01,
        epochs=2000,
        activation="logistic",
        stopping_threshold=0.0001,
        patience=5,
    ):

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(ACTIVATIONS.keys())}"
            )

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_name = activation
        self.stopping_threshold = stopping_threshold
        self.patience = patience if patience is not None else 1

        self.activation, self.activation_derivative = ACTIVATIONS[activation]

        self.weights = None
        self.bias = None

        self.errors = []
        self.epochs_run = 0

    def _initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

    def _forward(self, X):
        """
        Perform the forward pass.

        z = XW + b
        y_pred = activation(z)
        """

        z = np.dot(X, self.weights) + self.bias
        y_pred = self.activation(z)

        return z, y_pred

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        self._initialize_parameters(X.shape[1])

        self.errors = []
        self.epochs_run = 0
        n_samples = X.shape[0]

        previous_error = None
        patience_counter = 0

        for epoch in range(self.epochs):

            z, y_pred = self._forward(X)

            error = y_pred - y

            activation_gradient = self.activation_derivative(z)

            delta = error * activation_gradient

            gradient_weights = np.dot(X.T, delta) / n_samples
            gradient_bias = np.mean(delta)

            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

            mse = np.mean(error ** 2)
            self.errors.append(mse)

            if self.stopping_threshold is not None and previous_error is not None:
                change = abs(mse - previous_error)
                if change < self.stopping_threshold:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
                else:
                    patience_counter = 0

            previous_error = mse

        self.epochs_run = len(self.errors)
        return self

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        if self.weights is None:
            raise ValueError("Model has not been trained yet.")

        _, predictions = self._forward(X)

        return predictions
