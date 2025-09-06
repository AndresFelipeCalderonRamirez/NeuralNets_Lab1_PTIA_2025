import numpy as np
from abc import ABC, abstractmethod

class Cost(ABC):
    """
    Abstract class that defines the core behavior for cost functions (error functions) in a neural network.
    It represents the cost function (loss function) that measures the error between predicted and true values.

    This class uses the Factory Design Pattern to dynamically create cost function objects based on a string input.

    Methods:
        use(cls, name: str) -> "Cost":
            Given the name of a cost function, it returns an instance of the corresponding cost function class.

        value(self, y: np.ndarray, yp: np.ndarray) -> float:
            Computes the cost (error) between the true and predicted values.

        derivative(self, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
            Computes the derivative (gradient) of the cost function element-wise with respect to the predicted values.
    """

    # A dictionary to map cost function names (strings) to their respective class names
    cost_map = {
        'cross_entropy': 'CrossEntropy',
    }

    @classmethod
    def use(cls, name: str) -> "Cost":
        """
        Returns the cost function object based on the provided name.

        This method implements the Factory Design Pattern, dynamically creating and returning
        an instance of the corresponding cost function class (e.g., `MeanSquaredError`, `CrossEntropy`).

        Args:
            name (str): The name of the cost function (e.g., 'mse', 'cross_entropy').

        Returns:
            Cost: An instance of the corresponding cost function class.

        Raises:
            ValueError: If the provided name doesn't match any known cost functions.
        """
        cost_class = cls.cost_map.get(name.lower())
        if cost_class:
            return globals()[cost_class]()
        else:
            raise ValueError(f"Cost function '{name}' not supported")

    @abstractmethod
    def value(self, y: np.ndarray, yp: np.ndarray) -> float:
        """
        Computes the cost function value (error) between true values and predicted values.

        Args:
            y (np.ndarray): The true values (targets).
            yp (np.ndarray): The predicted values.
        Returns:
            float: The computed cost (error) between true and predicted values.
        """
        pass

    @abstractmethod
    def derivative(self, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Computes the derivative (gradient) of the cost function element-wise with respect to the predicted values.

        Args:
            y (np.ndarray): The true values (targets).
            yp (np.ndarray): The predicted values.
        Returns:
            np.ndarray: The derivative (gradient) of the cost function with respect to the predicted values.
        """
        pass