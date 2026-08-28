# fcnn.py
# Defines the generic multi-layer Fully Connected Neural Network used for both
# classification and regression in Assignment 2, trained via SGD backpropagation.
# New file — has no Assignment 1 equivalent (Assignment 1 only had a single-neuron Perceptron).

# class FCNN
#   __init__(layer_sizes, hidden_activation, output_activation, learning_rate, epochs, seed)
#       - Stores the network architecture (input/hidden/output sizes), activation function choices,
#         and training hyperparameters.
#
#   _initialize_parameters()
#       - Randomly initializes weights and biases for every layer (seeded for reproducibility) so
#         hidden nodes don't start symmetric and get stuck learning identical features.
#
#   _forward(X)
#       - Runs a forward pass through all layers and returns the pre-activation (z) and
#         post-activation (a) values at every layer, not just the final output.
#
#   _backward(activations, y)
#       - Computes per-layer gradients via chain-rule backpropagation, propagating error from the
#         output layer back through each hidden layer.
#
#   fit(X, y)
#       - Trains with true stochastic gradient descent: shuffles the sample order every epoch,
#         updates weights immediately after each individual sample, and logs the epoch-averaged
#         squared error for the error-vs-epoch plot.
#
#   predict(X)
#       - Returns final-layer predictions: argmax over output nodes for classification, raw output
#         value for regression.
#
#   forward_all(X)
#       - Returns the activation of every hidden and output node for a given input batch, used to
#         generate the per-node output plots required by the assignment.
