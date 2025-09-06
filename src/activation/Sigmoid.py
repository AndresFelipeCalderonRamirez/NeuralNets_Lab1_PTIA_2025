import numpy as np
from Activation import Activation

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    Implements the activation and its derivative for neural networks.

    Methods:
       - value(x: np.ndarray) -> np.ndarray: Computes the Sigmoid output for input x.
       - derivative(x: np.ndarray) -> np.ndarray: Computes the Sigmoid derivative for input x.
    """
    def __init__(self):
      super().__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the Sigmoid activation function.
        The Sigmoid function is defined as 1 / (1 + exp(-X)).

        Args:
        - x (np.ndarray): Array of input values.

        Returns:
        - np.ndarray: Result of applying the Sigmoid function to each element of X.
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the Sigmoid activation function.
        The derivative is computed as Sigmoid(x) * (1 - Sigmoid(x)).

        Args:
        - x (np.ndarray): Array of input values.

        Returns:
        - np.ndarray: Derivative of the Sigmoid function for each input value.
        """
        return np.exp(-x) / (1 + np.exp(-x)) ** 2