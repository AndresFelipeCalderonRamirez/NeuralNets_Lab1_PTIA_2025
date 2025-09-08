import numpy as np
from .Activation import Activation

class Relu(Activation):
    """
    RELU (Rectified Linear Unit) activation function.
    Implements the activation and its derivative for neural networks.

    Methods:
       - value(x: np.ndarray) -> np.ndarray: Computes the ReLU output for input X.
       - derivative(x: np.ndarray) -> np.ndarray: Computes the ReLU derivative for input X.
    """
    def __init__(self):
        super().__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the output of the ReLU activation function.
        For each input value, returns 0 if it is less than 0,
        otherwise, returns the input value as is.

        Args:
        - x (np.ndarray): Array of input values.

        Returns:
        - np.ndarray: Result of applying the ReLU function to each element of X.
        """
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the ReLU activation function.
        The derivative is 0 if the input value is less than 0,
        and 1 if it is greater than or equal to 0.

        Args:
        - x (np.ndarray): Array of input values.

        Returns:
        - np.ndarray: Derivative of ReLU for each input value.
        """
        return (x > 0).astype(np.float32)