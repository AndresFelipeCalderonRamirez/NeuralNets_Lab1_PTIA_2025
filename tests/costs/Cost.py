import unittest
import numpy as np
from src.costs import Cost

class TestCrossEntropy(unittest.TestCase):
    """
    Unit tests for the Cross-Entropy cost function.
    The Cross-Entropy loss is commonly used in classification tasks to measure
    the performance of a classification model whose output is a probability value.
    It quantifies the difference between two probability distributions: the true
    labels (`y`) and the predicted labels (`yp`).
    """

    def setUp(self):
        """
        Initializes the Cross-Entropy cost function before each test.
        """
        self.ce = Cost.use("cross_entropy")

    def test_value_basic(self):
        """
        Tests the basic functionality of the cross-entropy cost function.
        This test ensures the cross-entropy loss is calculated correctly
        for a simple example with true labels and predicted probabilities.

        The formula for cross-entropy is:
        H(y, yp) = -sum(y * log(yp)) / N
        where `y` is the true labels, `yp` is the predicted probabilities,
        and N is the number of samples.

        Expected value is calculated manually for the given example.
        """
        y = np.array([[1, 0], [0, 1]])
        yp = np.array([[0.9, 0.1], [0.2, 0.8]])
        expected = (-1) * (np.sum(y * np.log(yp)) / int(y.shape[0]))
        result = self.ce.value(y, yp)
        self.assertAlmostEqual(result, expected, places=5)

    def test_derivative_shape(self):
        """
        Tests that the derivative of the cross-entropy cost function has
        the same shape as the true label array `y`.

        The derivative should be calculated for each sample individually,
        and the shape of the result should match that of `y`.
        """
        y = np.array([[1, 0], [0, 1]])
        yp = np.array([[0.9, 0.1], [0.2, 0.8]])
        derivative = self.ce.derivative(y, yp)
        self.assertEqual(derivative.shape, y.shape)

    def test_value_shape_mismatch(self):
        """
        Tests that the `value` method raises a ValueError when the shapes of
        the true labels `y` and the predicted probabilities `yp` do not match.

        Cross-entropy requires the shapes of `y` and `yp` to be the same.
        """
        y = np.array([[1, 0]])
        yp = np.array([[0.9, 0.1], [0.2, 0.8]])
        with self.assertRaises(ValueError):
            self.ce.value(y, yp)

    def test_derivative_shape_mismatch(self):
        """
        Tests that the `derivative` method raises a ValueError when the shapes
        of the true labels `y` and the predicted probabilities `yp` do not match.

        The true labels and predicted probabilities must have the same shape
        to compute the derivative correctly.
        """
        y = np.array([[1, 0]])
        yp = np.array([[0.9, 0.1], [0.2, 0.8]])
        with self.assertRaises(ValueError):
            self.ce.derivative(y, yp)

    def test_derivative_invalid_probabilities(self):
        """
        Tests that the `derivative` method raises a ValueError when the predicted
        probabilities contain invalid values (e.g., exactly 0 or 1).

        The cross-entropy derivative requires valid probabilities in the range (0, 1).
        Probabilities exactly equal to 0 or 1 would cause issues when computing the log.
        """
        y = np.array([[1, 0], [0, 1]])
        yp = np.array([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            self.ce.derivative(y, yp)

if __name__ == '__main__':
    unittest.main()
