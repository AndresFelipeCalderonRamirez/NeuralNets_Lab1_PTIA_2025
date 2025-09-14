import unittest
import numpy as np
from src.activation.Activation import Activation

class TestRelu(unittest.TestCase):
    """
    Unit tests for the ReLU (Rectified Linear Unit) activation function.
    The ReLU activation function outputs the input directly if it is positive,
    otherwise, it will output zero. This is commonly used in deep learning models
    to introduce non-linearity.

    Mathematically:
    ReLU(x) = max(0, x)

    This class tests:
    - The behavior of ReLU for positive and negative inputs.
    - The derivative of ReLU.
    """

    def setUp(self):
        """
        Initializes the ReLU activation function before each test.
        """
        self.relu = Activation.use("relu")

    def test_value_positive(self):
        """
        Tests that ReLU returns the same values for positive inputs.
        For inputs greater than zero, the output should be the input itself.

        Example:
        If x = [1.0, 2.0, 3.0], the output should be [1.0, 2.0, 3.0]
        """
        x = np.array([1.0, 2.0, 3.0])
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(self.relu.value(x), expected)

    def test_value_negative(self):
        """
        Tests that ReLU returns zero for negative and zero inputs.
        For inputs less than or equal to zero, the output should be zero.

        Example:
        If x = [-1.0, -2.0, 0.0], the output should be [0.0, 0.0, 0.0]
        """
        x = np.array([-1.0, -2.0, 0.0])
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(self.relu.value(x), expected)

    def test_derivative(self):
        """
        Tests the derivative of the ReLU function:
        - 0 for x <= 0
        - 1 for x > 0

        The derivative of ReLU is piecewise:
        d(ReLU(x))/dx = 0 for x <= 0, and d(ReLU(x))/dx = 1 for x > 0

        Example:
        If x = [-1.0, 0.0, 2.0], the derivative should be [0.0, 0.0, 1.0]
        """
        x = np.array([-1.0, 0.0, 2.0])
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(self.relu.derivative(x), expected)

class TestSigmoid(unittest.TestCase):
    """
    Unit tests for the Sigmoid activation function.
    The Sigmoid function maps any real-valued number into the range (0, 1),
    making it suitable for binary classification tasks. It is defined as:

    Sigmoid(x) = 1 / (1 + exp(-x))

    This class tests:
    - The behavior of Sigmoid at specific values, particularly at 0.
    - The output range of Sigmoid for extreme inputs.
    - The derivative of Sigmoid, particularly at x = 0.
    """

    def setUp(self):
        """
        Initializes the Sigmoid activation function before each test.
        """
        self.sigmoid = Activation.use("sigmoid")

    def test_value(self):
        """
        Tests that sigmoid(0) returns 0.5, as the Sigmoid function has a value of 0.5
        when the input is zero.

        Example:
        Sigmoid(0) = 1 / (1 + exp(0)) = 0.5
        """
        x = np.array([0.0])
        expected = np.array([0.5])
        np.testing.assert_almost_equal(self.sigmoid.value(x), expected, decimal=5)

    def test_value_range(self):
        """
        Tests that the sigmoid output is always in the range [0, 1],
        even for extreme input values. Sigmoid function should saturate at
        both ends of the range for very large or very small inputs.

        Example:
        Sigmoid(-100) ≈ 0, Sigmoid(100) ≈ 1
        """
        x = np.array([-100.0, 0.0, 100.0])
        result = self.sigmoid.value(x)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))

    def test_derivative(self):
        """
        Tests the derivative of the Sigmoid function at x = 0.
        The derivative of Sigmoid(x) is given by:
        Sigmoid'(x) = Sigmoid(x) * (1 - Sigmoid(x))

        At x = 0, Sigmoid(0) = 0.5, so the derivative should be 0.25.

        Example:
        Sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
        """
        x = np.array([0.0])
        expected = np.array([0.25])
        np.testing.assert_almost_equal(self.sigmoid.derivative(x), expected, decimal=5)

if __name__ == '__main__':
    unittest.main()
