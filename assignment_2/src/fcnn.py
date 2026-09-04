"""
fcnn.py

Fully Connected Neural Network (FCNN) for Assignment 2.

The network is implemented from scratch using NumPy and follows the
FCNN / backpropagation formulation taught in class.

Supported tasks:
    - Classification
    - Regression

Network architecture:
    Input -> Hidden layer(s) -> Output

Input layer:
    Linear

Hidden layers:
    Sigmoidal (logistic or tanh)

Output layer:
    Sigmoidal (logistic or tanh) for classification
    Linear for regression

Training:
    - Pattern mode
    - Stochastic Gradient Descent (SGD)
    - Backpropagation
    - Squared instantaneous error
    - Samples shuffled at the beginning of every epoch

Bias:
    Bias is represented using an additional input/node whose value is 1.
    Therefore, for a layer with N neurons connected to M neurons, the
    corresponding weight matrix has shape:

        (N + 1, M)

    The first row contains the bias weights.

No external machine-learning or neural-network libraries are used.
"""

import numpy as np

from utils.activations import ACTIVATIONS


class FCNN:
    """
    Fully Connected Neural Network trained using SGD backpropagation.

    Parameters
    ----------
    layer_sizes : list or tuple
        Number of neurons in each layer, including input and output.

        Examples:
            [2, 8, 3]
                2 input neurons
                8 hidden neurons
                3 output neurons

            [2, 20, 10, 3]
                2 input neurons
                20 neurons in hidden layer 1
                10 neurons in hidden layer 2
                3 output neurons

            [1, 8, 1]
                1 input neuron
                8 hidden neurons
                1 output neuron

    hidden_activation : str
        Activation function used by all hidden layers.
        Supported:
            "logistic"
            "tanh"

    output_activation : str
        Activation function used by the output layer.
        Supported:
            "logistic"
            "tanh"
            "linear"

    learning_rate : float
        Learning rate eta. Class notes specify 0 < eta <= 1.

    epochs : int
        Maximum number of training epochs.

    seed : int or None
        Seed for reproducible initialization and shuffling.

    stopping_threshold : float or None
        If supplied, training stops when the absolute difference
        between average errors of successive epochs becomes smaller
        than this threshold.
    """

    def __init__(
        self,
        layer_sizes,
        hidden_activation="logistic",
        output_activation="logistic",
        learning_rate=0.01,
        epochs=1000,
        seed=None,
        stopping_threshold=None,
    ):

        if not isinstance(layer_sizes, (list, tuple)):
            raise ValueError("layer_sizes must be a list or tuple.")

        if len(layer_sizes) < 3:
            raise ValueError(
                "FCNN must contain an input layer, at least one "
                "hidden layer, and an output layer."
            )

        if any(
            not isinstance(size, (int, np.integer)) or size <= 0
            for size in layer_sizes
        ):
            raise ValueError(
                "Every value in layer_sizes must be a positive integer."
            )

        if hidden_activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown hidden activation '{hidden_activation}'. "
                f"Choose from {list(ACTIVATIONS.keys())}"
            )

        if output_activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown output activation '{output_activation}'. "
                f"Choose from {list(ACTIVATIONS.keys())}"
            )

        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError(
                "learning_rate must satisfy 0 < learning_rate <= 1."
            )

        if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")

        if stopping_threshold is not None and stopping_threshold < 0:
            raise ValueError(
                "stopping_threshold must be non-negative or None."
            )


        self.layer_sizes = list(layer_sizes)

        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.seed = seed
        self.stopping_threshold = stopping_threshold

        
        (
            self.hidden_activation,
            self.hidden_activation_derivative,
        ) = ACTIVATIONS[hidden_activation]

        (
            self.output_activation,
            self.output_activation_derivative,
        ) = ACTIVATIONS[output_activation]

        # Linear output -> regression
        # Sigmoidal output -> classification
        if output_activation in ("linear", "identity"):
            self.task_type = "regression"
        else:
            self.task_type = "classification"

        
        # Parameters
        #
        # weights[l] connects layer l to layer l+1.
        #
        # Shape:
        #     (number of neurons in previous layer + 1,
        #      number of neurons in current layer)
        #
        # Row 0 contains bias weights.

        self.weights = []


        self.errors = []

        self.rng = np.random.default_rng(seed)

        self.is_fitted = False

        self._initialize_parameters()


    def _initialize_parameters(self):
        """
        Initialize every weight matrix with random values.

        For a connection from a layer with N neurons to a layer with M
        neurons:

            weight matrix shape = (N + 1, M)

        The first row corresponds to bias weights.

        Values are drawn from a zero-mean normal distribution with
        standard deviation:

            1 / sqrt(N)

        where N is the number of actual neurons in the previous layer.
        """

        self.weights = []

        for layer_index in range(len(self.layer_sizes) - 1):

            n_inputs = self.layer_sizes[layer_index]
            n_outputs = self.layer_sizes[layer_index + 1]

            # +1 for the bias row.
            shape = (n_inputs + 1, n_outputs)

            # Scale according to the number of neurons in the
            # previous layer.
            scale = 1.0 / np.sqrt(n_inputs)

            weight_matrix = self.rng.normal(
                loc=0.0,
                scale=scale,
                size=shape,
            )

            self.weights.append(weight_matrix)

   
    # Bias handling


    @staticmethod
    def _add_bias(vector):
        """
        Add the bias input/node.

        [x1, x2, ..., xd]

        becomes

        [1, x1, x2, ..., xd]
        """

        vector = np.asarray(vector, dtype=float).reshape(-1)

        return np.concatenate(
            (
                np.array([1.0]),
                vector,
            )
        )

   

    def _forward_single(self, x):
        """
        Perform forward computation for one example.

        For each layer:

            a = W^T x_augmented

        followed by:

            activation = f(a)

        Returns
        -------
        pre_activations : list
            Weighted sums for every non-input layer.

        activations : list
            Outputs of every layer, including the input layer.
        """

        x = np.asarray(x, dtype=float).reshape(-1)

        if x.shape[0] != self.layer_sizes[0]:
            raise ValueError(
                f"Expected {self.layer_sizes[0]} input features, "
                f"but received {x.shape[0]}."
            )

        # Input layer is linear.
        current_activation = x

        activations = [current_activation.copy()]
        pre_activations = []

        for layer_index, weight_matrix in enumerate(self.weights):

            # Add bias input.
            augmented_activation = self._add_bias(
                current_activation
            )

            # Weighted sum:
            #
            # a = W^T x
            
            z = np.dot(
                weight_matrix.T,
                augmented_activation,
            )

            pre_activations.append(z)

            # Output layer uses its own activation.
            if layer_index == len(self.weights) - 1:
                current_activation = self.output_activation(z)

            # All hidden layers use hidden activation.
            else:
                current_activation = self.hidden_activation(z)

            activations.append(
                np.asarray(current_activation, dtype=float).copy()
            )

        return pre_activations, activations


    def _forward(self, X):
        """
        Perform forward computation for all examples.

        Training uses _forward_single() because training is pattern-mode
        SGD. This batch version is useful for evaluation and plotting.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Expected {self.layer_sizes[0]} input features, "
                f"but received {X.shape[1]}."
            )

        batch_pre_activations = [
            [] for _ in range(len(self.weights))
        ]

        batch_activations = [
            [] for _ in range(len(self.layer_sizes))
        ]

        for x in X:

            pre_activations, activations = self._forward_single(x)

            for layer_index in range(len(self.weights)):
                batch_pre_activations[layer_index].append(
                    pre_activations[layer_index]
                )

            for layer_index in range(len(self.layer_sizes)):
                batch_activations[layer_index].append(
                    activations[layer_index]
                )

        batch_pre_activations = [
            np.asarray(values, dtype=float)
            for values in batch_pre_activations
        ]

        batch_activations = [
            np.asarray(values, dtype=float)
            for values in batch_activations
        ]

        return batch_pre_activations, batch_activations


    def _backward(self, pre_activations, activations, y):
        """
        Perform backpropagation for ONE training example.

        Instantaneous error:

            E_n = 1/2 * sum_k (y_nk - y_hat_nk)^2

        Output delta:

            delta_k = (y_hat_k - y_k) f'(a_k)

        Hidden delta:

            delta_j = f'(a_j)
                       * sum_k(delta_k * w_jk)

        Weight gradient:

            dE/dw_jk = delta_k * previous_activation_j

        Returns
        -------
        gradients : list
            Gradient of the instantaneous error with respect to every
            weight matrix.
        """

        y = np.asarray(y, dtype=float).reshape(-1)

        if y.shape[0] != self.layer_sizes[-1]:
            raise ValueError(
                f"Expected {self.layer_sizes[-1]} target values, "
                f"but received {y.shape[0]}."
            )

        n_layers = len(self.weights)

        # One delta vector for every non-input layer.
        deltas = [None] * n_layers        

        output = activations[-1]

        output_derivative = self.output_activation_derivative(
            pre_activations[-1]
        )

        deltas[-1] = (
            (output - y)
            * output_derivative
        )

        # Hidden layer deltas
        #
        # Work backwards:
        #
        # output -> last hidden -> ... -> first hidden

        for layer_index in range(n_layers - 2, -1, -1):

            hidden_derivative = self.hidden_activation_derivative(
                pre_activations[layer_index]
            )

            # Row 0 of the next weight matrix is the bias row.
            # Bias is NOT propagated as a neuron to the previous layer.
            next_weights_without_bias = self.weights[
                layer_index + 1
            ][1:, :]

            deltas[layer_index] = (
                hidden_derivative
                * np.dot(
                    next_weights_without_bias,
                    deltas[layer_index + 1],
                )
            )

        
        # Weight gradients

        gradients = []

        for layer_index in range(n_layers):

            previous_activation = activations[layer_index]

            # Include bias as the first input.
            previous_activation_augmented = self._add_bias(
                previous_activation
            )

            # Outer product gives all weight derivatives for this layer.
            gradient = np.outer(
                previous_activation_augmented,
                deltas[layer_index],
            )

            gradients.append(gradient)

        return gradients


    @staticmethod
    def _instantaneous_error(y, y_pred):
        """
        Squared instantaneous error:

            E_n = 1/2 * sum_k (y_nk - y_hat_nk)^2
        """

        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        return 0.5 * np.sum(
            (y - y_pred) ** 2
        )


    def _update_weights(self, gradients):
        """
        Immediate pattern-mode SGD update:

            W_new = W_old - eta * dE_n/dW
        """

        for layer_index in range(len(self.weights)):

            self.weights[layer_index] -= (
                self.learning_rate
                * gradients[layer_index]
            )

    
    # Training

    def fit(self, X, y):
        """
        Train the FCNN using pattern-mode stochastic gradient descent.

        For every epoch:

            1. Shuffle all training examples.
            2. Present one example.
            3. Forward computation.
            4. Calculate instantaneous error.
            5. Backpropagate.
            6. Immediately update weights.
            7. Repeat until every example has been presented.
            8. Calculate average epoch error.
            9. Check stopping criterion.

        Returns
        -------
        self
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:

            if self.layer_sizes[0] != 1:
                raise ValueError(
                    "X must be 2-dimensional for more than one input "
                    "feature."
                )

            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )


        if y.ndim == 1:

            if self.layer_sizes[-1] == 1:
                y = y.reshape(-1, 1)

            elif self.task_type == "classification":
                y = self._encode_class_labels(y)

            else:
                y = y.reshape(-1, 1)

        elif y.ndim != 2:
            raise ValueError(
                "y must be a 1-dimensional or 2-dimensional array."
            )


        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )


        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Expected {self.layer_sizes[0]} input features, "
                f"but received {X.shape[1]}."
            )

        if y.shape[1] != self.layer_sizes[-1]:
            raise ValueError(
                f"Expected {self.layer_sizes[-1]} output values, "
                f"but received {y.shape[1]}."
            )

        n_samples = X.shape[0]

        if n_samples == 0:
            raise ValueError(
                "Training data cannot be empty."
            )


        self.errors = []
        self.is_fitted = False

        previous_average_error = None


        for epoch in range(self.epochs):

            # Shuffle presentation order every epoch.
            indices = self.rng.permutation(n_samples)

            total_error = 0.0

            # Pattern-mode SGD

            for index in indices:

                x_n = X[index]
                y_n = y[index]

                # Forward computation for ONE example.
                pre_activations, activations = (
                    self._forward_single(x_n)
                )

                y_pred = activations[-1]

                # Instantaneous squared error.
                error = self._instantaneous_error(
                    y_n,
                    y_pred,
                )

                total_error += error

                # Backward computation for ONE example.
                gradients = self._backward(
                    pre_activations,
                    activations,
                    y_n,
                )

                # Immediate weight update.
                self._update_weights(gradients)

           
            # Average error for the completed epoch

            average_error = total_error / n_samples

            self.errors.append(average_error)

            # Stopping criterion
            #
            # |E_av(m) - E_av(m-1)| < threshold

            if (
                self.stopping_threshold is not None
                and previous_average_error is not None
            ):

                change_in_error = abs(
                    average_error
                    - previous_average_error
                )

                if change_in_error < self.stopping_threshold:
                    break

            previous_average_error = average_error

        self.is_fitted = True

        return self


    def _encode_class_labels(self, y):
        """
        Convert integer class labels into output vectors.

        Logistic output:
            class 0 -> [1, 0, 0, ...]
            class 1 -> [0, 1, 0, ...]
            ...

        Tanh output:
            class 0 -> [ 1, -1, -1, ...]
            class 1 -> [-1,  1, -1, ...]
            ...

        Class labels are assumed to be:

            0, 1, ..., K-1
        """

        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError(
                "Class labels must be a 1-dimensional array."
            )

        if not np.all(np.equal(y, y.astype(int))):
            raise ValueError(
                "Class labels must contain integer values."
            )

        y = y.astype(int)

        n_classes = self.layer_sizes[-1]

        if np.any(y < 0) or np.any(y >= n_classes):
            raise ValueError(
                f"Class labels must be in the range "
                f"0 to {n_classes - 1}."
            )

        if self.output_activation_name == "tanh":

            encoded = -np.ones(
                (len(y), n_classes),
                dtype=float,
            )

            encoded[
                np.arange(len(y)),
                y,
            ] = 1.0

        else:

            encoded = np.zeros(
                (len(y), n_classes),
                dtype=float,
            )

            encoded[
                np.arange(len(y)),
                y,
            ] = 1.0

        return encoded

    def predict(self, X):
        """
        Generate predictions.

        Classification:
            Returns argmax of output neurons.

        Regression:
            Returns raw output values.
        """

        if not self.is_fitted:
            raise ValueError(
                "Model has not been trained yet."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:

            if self.layer_sizes[0] == 1:
                X = X.reshape(-1, 1)
            else:
                X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        _, activations = self._forward(X)

        outputs = activations[-1]

        # Classification:
        # class = argmax(output neurons)
        if self.task_type == "classification":

            return np.argmax(
                outputs,
                axis=1,
            )

        # Regression:
        # return continuous output.
        if self.layer_sizes[-1] == 1:

            return outputs.reshape(-1)

        return outputs


    def predict_output(self, X):
        """
        Return raw output-layer values.

        Useful for:
            - regression
            - classification analysis
            - decision-region plots
            - output-node plots
        """

        if not self.is_fitted:
            raise ValueError(
                "Model has not been trained yet."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:

            if self.layer_sizes[0] == 1:
                X = X.reshape(-1, 1)
            else:
                X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        _, activations = self._forward(X)

        return activations[-1]

    def forward_all(self, X):
        """
        Return activations of every layer.

        For:

            [2, 8, 3]

        returns:

            activations[0] -> input layer
            activations[1] -> hidden layer
            activations[2] -> output layer

        For:

            [2, 20, 10, 3]

        returns:

            activations[0] -> input
            activations[1] -> hidden layer 1
            activations[2] -> hidden layer 2
            activations[3] -> output

        This is used for the per-node output plots required by
        Assignment 2.
        """

        if not self.is_fitted:
            raise ValueError(
                "Model has not been trained yet."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:

            if self.layer_sizes[0] == 1:
                X = X.reshape(-1, 1)
            else:
                X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        _, activations = self._forward(X)

        return activations


    def forward_details(self, X):
        """
        Return both pre-activation and post-activation values.

        Returns:

            pre_activations, activations
        """

        if not self.is_fitted:
            raise ValueError(
                "Model has not been trained yet."
            )

        return self._forward(X)


    def get_weights(self):
        """
        Return copies of the trained weight matrices.

        The first row of every matrix contains bias weights.
        """

        return [
            weight_matrix.copy()
            for weight_matrix in self.weights
        ]