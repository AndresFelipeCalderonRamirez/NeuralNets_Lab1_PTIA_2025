import unittest
import numpy as np
from src.dense import DenseANN

class TestDenseANN(unittest.TestCase):
    """
    Unit tests for the DenseANN class, covering various logical gates and custom datasets.
    The tests aim to validate the functionality of the neural network for different tasks,
    including AND, OR, XOR, and a custom dataset.

    Tests:
        - `test_and`: Verifies if the network can learn the AND function.
        - `test_or`: Verifies if the network can learn the OR function.
        - `test_xor_perceptron`: Verifies that a single-layer perceptron cannot learn XOR.
        - `test_xor_multilayer`: Verifies if a multi-layer perceptron can learn XOR.
        - `test_custom_dataset`: Tests the network on a custom dataset with random binary inputs.
    """

    def test_and(self):
        """
        Verifies if the neural network can learn the AND logical function.

        X (ndarray): Input array with 4 samples (XOR truth table).
        Y (ndarray): Expected output array (0, 0, 0, 1).
        """
        x = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([[0],[0],[0],[1]])

        model = DenseANN([2,1], learning_rate=0.1)
        model.train(x, y, epochs=2000, print_cost=False, do_graphic=False)

        preds = np.round(model.predict(x))
        np.testing.assert_array_equal(preds, y, "The perceptron did not learn AND correctly")

    def test_or(self):
        """
        Verifies if the neural network can learn the OR logical function.

        X (ndarray): Input array with 4 samples (OR truth table).
        Y (ndarray): Expected output array (0, 1, 1, 1).
        """
        x = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([[0],[1],[1],[1]])

        model = DenseANN([2,1], learning_rate=0.1)
        model.train(x, y, epochs=2000, print_cost=False, do_graphic=False)

        preds = np.round(model.predict(x))
        np.testing.assert_array_equal(preds, y, "The perceptron did not learn OR correctly")

    def test_xor_perceptron(self):
        """
        Verifies that a single-layer perceptron cannot learn the XOR function.

        X (ndarray): Input array with 4 samples (XOR truth table).
        Y (ndarray): Expected output array (0, 1, 1, 0).
        """
        x = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([[0],[1],[1],[0]])

        model = DenseANN([2,1], learning_rate=0.1)
        model.train(x, y, epochs=3000, print_cost=False, do_graphic=False)

        preds = np.round(model.predict(x))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(preds, y)

    def test_xor_multilayer(self):
        """
        Verifies if a multi-layer perceptron can successfully learn the XOR function.

        X (ndarray): Input array with 4 samples (XOR truth table).
        Y (ndarray): Expected output array (0, 1, 1, 0).
        """
        x = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([[0],[1],[1],[0]])

        model = DenseANN([2, 4, 1], learning_rate=0.05)
        model.train(x, y, epochs=10000, print_cost=False, do_graphic=False)

        preds = np.round(model.predict(x))
        np.testing.assert_array_equal(preds, y, "The multi-layer network did not learn XOR correctly")

    def test_custom_dataset(self):
        """
        Tests the neural network on a custom random binary dataset.

        Generates a dataset with 100 samples, each having 3 binary features.
        The target is a binary classification task, where the output is the parity
        (even or odd) of the sum of the input values.

        X_train, X_test: Split dataset into training and test sets (80%-20%).
        Y_train, Y_test: Corresponding target labels.
        """
        rng = np.random.default_rng(seed=42)
        x = rng.integers(0, 2, size=(100, 3))
        y = (np.sum(x, axis=1) % 2).reshape(-1, 1)

        m = x.shape[0]
        split = int(0.8 * m)

        indices = rng.permutation(m)

        train_idx, test_idx = indices[:split], indices[split:]

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = DenseANN([x.shape[1], 5, 1], learning_rate=0.05)
        model.train(x_train, y_train, epochs=2000, print_cost=False, do_graphic=False)

        preds = np.round(model.predict(x_test))
        accuracy = np.mean(preds == y_test)

        self.assertGreaterEqual(accuracy, 0.7, "The network did not reach at least 70% accuracy on the dataset")

if __name__ == '__main__':
    unittest.main()
