import numpy as np
from ..activation.Activation import Activation

class DenseLayer:
    """ Represents a layer (hidden or output) in the neural network """

    def __init__(self, input_size: int, neurons: int, activation: str, learning_rate: float):
        """ Initializes a layer of neurons within the neural network.

        Args:
            input_size (int): number of neurons in the previous layer or input features
            neurons (int): number of neurons in the layer
            activation (str): name of the activation function
            learning_rate (float): eta, learning rate of the network
        """
        self.learning_rate = learning_rate
        self.activation = Activation.use(activation)

        seed = 42
        rng = np.random.default_rng(seed)
        if activation == "relu":
            limit = np.sqrt(2 / input_size)
            self.W = rng.normal(0, limit, size=(input_size, neurons))
        else:
            limit = np.sqrt(1 / input_size)
            self.W = rng.normal(0, limit, size=(input_size, neurons))

        self.b = np.zeros((1, neurons))
        self.d_w = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, a_b: np.ndarray):
        """ Forward pass: compute activation values for the layer.

        Args:
            a_b (np.ndarray): activations (or input) from the previous layer.
        Returns:
            np.ndarray: activation values of the current layer.
        """
        self.a_prev = a_b
        self.Z = np.dot(a_b, self.W) + self.b
        self.A = self.activation.value(self.Z)
        return self.A

    def backward(self, a_b: np.ndarray):
        """ Backward pass: compute gradients with respect to weights and biases.

        Args:
            a_b (np.ndarray): gradients (or errors) from the next layer.
        Returns:
            np.ndarray: gradients with respect to the input of the current layer.
        """
        m = self.a_prev.shape[0]

        d_z = a_b * self.activation.derivative(self.Z)
        self.d_w = (1 / m) * np.dot(self.a_prev.T, d_z)
        self.db = (1 / m) * np.sum(d_z, axis=0, keepdims=True)
        d_a_prev = np.dot(d_z, self.W.T)

        return d_a_prev

    def update_parameters(self):
        """ Update the parameters (weights and biases) using the gradients. """
        self.W -= self.learning_rate * self.d_w
        self.b -= self.learning_rate * self.db

    def shapes(self):
        """ Returns the shapes of weights and biases.

        Returns:
            tuple: The shapes of W and b as a tuple (W.shape, b.shape).
        """
        return self.W.shape, self.b.shape

    def to_string(self, detail: bool):
        """ Returns a string representation of the layer.

        Args:
        detail (bool): Whether to include detailed information (like activation function).
        Returns:
        str: A string representation of the layer's configuration.
        """
        if detail:
            return f"Layer(input_size={self.W.shape[0]}, neurons={self.W.shape[1]}, activation={self.activation.__class__.__name__})"
        return f"Layer({self.W.shape[0]} -> {self.W.shape[1]})"