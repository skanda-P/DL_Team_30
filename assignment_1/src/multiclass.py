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

        if model.activation_name == "logistic":
            return 0.5

        elif model.activation_name == "tanh":
            return 0.0

        else:
            raise ValueError(
                f"Unsupported activation for classification: "
                f"{model.activation_name}"
            )

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.classifiers = {}

        # Generate all class pairs
        for class_a, class_b in combinations(self.classes_, 2):

            # Keep only samples belonging to the current pair
            mask = (y == class_a) | (y == class_b)

            X_pair = X[mask]
            y_pair_raw = y[mask]


            if self.activation == "logistic":
                # Logistic output range: (0, 1)
                #
                # class_a -> 0
                # class_b -> 1
                y_pair = np.where(
                    y_pair_raw == class_a,
                    0.0,
                    1.0
                )

            elif self.activation == "tanh":
                # Tanh output range: (-1, 1)
                #
                # class_a -> -1
                # class_b -> +1
                y_pair = np.where(
                    y_pair_raw == class_a,
                    -1.0,
                    1.0
                )

            else:
                raise ValueError(
                    f"Unsupported activation for classification: "
                    f"{self.activation}"
                )

            # Create binary perceptron
            model = Perceptron(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                activation=self.activation
            )

            # Train on this class pair
            model.fit(X_pair, y_pair)

            # Store model
            self.classifiers[(class_a, class_b)] = model

        return self

    def predict_pair_labels(self, class_a, class_b, X):

        X = np.asarray(X, dtype=float)

        model = self.classifiers[(class_a, class_b)]

        threshold = self._decision_threshold(model)

        raw = model.predict(X)

        # Convert continuous output into binary prediction
        hard = (raw >= threshold).astype(int)

        # Convert binary prediction back to original class labels
        return np.where(
            hard == 0,
            class_a,
            class_b
        )

    def predict(self, X):

        if self.classes_ is None:
            raise ValueError(
                "Model has not been trained yet."
            )

        X = np.asarray(X, dtype=float)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Map class label -> column index
        class_index = {
            c: i for i, c in enumerate(self.classes_)
        }

        # votes[sample, class]
        votes = np.zeros(
            (n_samples, n_classes),
            dtype=int
        )

        # Each pairwise classifier gets one vote
        for (class_a, class_b), model in self.classifiers.items():

            threshold = self._decision_threshold(model)

            raw = model.predict(X)

            hard = (raw >= threshold).astype(int)

            # Prediction = class_a
            votes[
                hard == 0,
                class_index[class_a]
            ] += 1

            # Prediction = class_b
            votes[
                hard == 1,
                class_index[class_b]
            ] += 1

        # Class with the most votes wins
        winners = np.argmax(votes, axis=1)

        return self.classes_[winners]