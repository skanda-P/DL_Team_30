# Activation functions
"""
Activation functions for the Deep Learning assignments.

This module contains activation functions and their derivatives.
The functions are implemented from scratch using NumPy.

Activations required for Assignment 1:
    - Sigmoid / Logistic
    - Hyperbolic Tangent (tanh)
    - Linear

"""

import numpy as np


def linear(z):
    """
    Linear (identity) activation function.
    f(z) = z
    """
    return np.asarray(z, dtype=float)


def linear_derivative(z):
    """
    Derivative of the linear activation function.
    f'(z) = 1
    """
    return np.ones_like(z, dtype=float)


def logistic(z):
    """
    Sigmoid (logistic) activation function.
    f(z) = 1 / (1 + exp(-z))
    Output range:
        (0, 1)
    """

    z = np.asarray(z, dtype=float)

    # Numerically stable implementation.
    # Directly computing exp(-z) can overflow for large negative values of z.
    output = np.empty_like(z)

    positive = z >= 0
    negative = ~positive

    # z >= 0
    output[positive] = 1.0 / (
        1.0 + np.exp(-z[positive])
    )

    # z < 0
    exp_z = np.exp(z[negative])
    output[negative] = exp_z / (
        1.0 + exp_z
    )

    return output


def logistic_derivative(z):
    """
    Derivative of the logistic activation.
    logistic'(z) = logistic(z) * (1 - logistic(z))
    """

    s = logistic(z)

    return s * (1.0 - s)



def logistic_derivative_from_output(output):
    """
    Compute the logistic derivative when logistic(z) has
    already been calculated.

    If:
        output = logistic(z)

    then:
        logistic'(z) = output * (1 - output)

    This avoids recomputing logistic(z).
    """

    return output * (1.0 - output)


def tanh(z):
    """
    Hyperbolic tangent activation function.

    f(z) = tanh(z)

    Output range:
        (-1, 1)

    """

    return np.tanh(z)


def tanh_derivative(z):
    """
    Derivative of the hyperbolic tangent activation.

    tanh'(z) = 1 - tanh(z)^2
    """

    t = np.tanh(z)

    return 1.0 - t**2


def tanh_derivative_from_output(output):
    """
    Compute tanh derivative when tanh(z) has already
    been calculated.

    If:
        output = tanh(z)

    then:
        tanh'(z) = 1 - output^2
    """

    return 1.0 - output**2


ACTIVATIONS = {
    "linear": (linear, linear_derivative),
    "identity": (linear, linear_derivative),

    "logistic": (logistic, logistic_derivative),

    "tanh": (tanh, tanh_derivative),
   
}