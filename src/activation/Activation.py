import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    """
    Abstract class that defines the core behavior for activation functions.
    It represents the activation function of any neuron in a neural network.

    Attributes:
        activation_map (dict): A dictionary mapping activation function names to their respective class names.

    Methods:
        use(cls, name: str) -> "Activation":
            Given the name of an activation function, it returns an instance of the corresponding activation function class.

        value(self, x: np.ndarray) -> np.ndarray:
            Computes the activation function element-wise on the input.

        derivative(self, x: np.ndarray) -> np.ndarray:
            Computes the derivative (gradient) of the activation function element-wise on the input.
    """

    activation_map = {}

    @classmethod
    def use(cls, name: str) -> "Activation":
        """
        Returns the activation function object based on the provided name.

        This method implements the Factory Design Pattern, where the instantiation of the
        appropriate activation function class is delegated based on the input name.
        The method dynamically creates and returns an instance of the corresponding activation
        function class (e.g., `Sigmoid`, `Relu`).

        Args:
            name (str): The name of the activation function (e.g., 'sigmoid', 'relu').
        Returns:
            Activation: An instance of the corresponding activation function class (e.g., `Sigmoid`, `Relu`).
        Raises:
            ValueError: If the provided name doesn't match any known activation functions.
        """
        activation_class = cls.activation_map.get(name.lower())
        if activation_class:
            return activation_class()
        else:
            raise ValueError(f"Activation '{name}' not supported")

    @abstractmethod
    def value(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the activation function element-wise on the input.
        Args:
            x (np.ndarray): The input array of values.
        Returns:
            np.ndarray: The output after applying the activation function element-wise.
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative (gradient) of the activation function element-wise on the input.
        Args:
            x (np.ndarray): The input array of values.
        Returns:
            np.ndarray: The derivative of the activation function for each element in the input.
        """
        pass