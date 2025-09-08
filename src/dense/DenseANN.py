import numpy as np
from .DenseLayer import DenseLayer
from ..costs.Cost import Cost
from ..metrics.Metric import Metric
import matplotlib.pyplot as plt

class DenseANN:
    """
    Represents a fully connected neural network.
    """
    def __init__(self, layers: list, learning_rate: float):
        """
        Initializes the layers, then stores the network architecture and learning rate.

        Args:
            layers (list): list of layer specifications, where:
                - list[0] is the number of input features,
                - list[1 to -1] are the neurons in each hidden layer,
                - list[-1] is the number of output neurons.
            learning_rate (float): learning rate (eta) of the network.
        """
        self.learning_rate = learning_rate
        self.layers = []

        for i in range(1, len(layers)):
            activation = "sigmoid" if i == len(layers) - 1 else "relu"
            self.layers.append(DenseLayer(layers[i - 1], layers[i], activation, self.learning_rate))

    def predict(self, x: np.ndarray):
        """
        Computes prediction values from the inputs.

        Args:
            x (ndarray): feature values (input).
        Returns:
            Yp (ndarray): predicted output values.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray):
        """
        Performs a forward pass through the network to generate predictions.

        Args:
            x (ndarray): feature values (input).
        Returns:
            Yp (ndarray): predicted output values.
        """
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, d_a: np.ndarray):
        """
        Computes the backward pass to calculate errors and gradients.

        Args:
        dA (ndarray): gradient values from the prediction.
        Returns:
        G (ndarray): network gradients.
        """
        d_a_prev = d_a
        for layer in reversed(self.layers):
            d_a_prev = layer.backward(d_a_prev)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, print_cost: bool, do_graphic: bool):
        """
        Trains the neural network.

        Args:
        X (ndarray): feature values - training set.
        Y (ndarray): expected output values - training set.
        epochs (int): number of iterations.
        print_cost (bool): whether to print the cost at each iteration.
        do_graphic (bool): whether to plot the cost evolution.
        """
        costs = []
        for epoch in range(epochs):
            yp = self.forward(x)

            cross_entropy = Cost.use("cross_entropy")

            cost = cross_entropy.value(y, yp)
            costs.append(cost)

            d_a = cross_entropy.derivative(y, yp)
            self.backward(d_a)

            for layer in self.layers:
                layer.update_parameters()

            accuracy = Metric.use("accuracy")
            metric_value = accuracy.value(y, yp)
            if print_cost and epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost}, Metric: {metric_value}")

        if do_graphic:
            plt.plot(costs)
            plt.xlabel("Epochs")
            plt.ylabel("Cost")
            plt.title("Cost Evolution")
            plt.show()

    def shapes(self):
        """
        Returns the sizes associated with the network.

        Returns:
            s (tuple<int>): a tuple of the network's layer sizes.
        """
        return tuple(layer.W.shape[0] for layer in self.layers)

    def to_string(self):
        """
        Returns a string representation of the network architecture.

        Returns:
            str: architecture as a string.
        """
        arch = " -> ".join([str(layer.W.shape[1]) for layer in self.layers] + [str(self.layers[-1].W.shape[0])])
        return f"DenseANN({arch})"