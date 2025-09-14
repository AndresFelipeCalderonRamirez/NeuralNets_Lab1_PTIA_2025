import numpy as np
from .Cost import Cost

class CrossEntropy(Cost):
    """
    Cross-Entropy Cost Function. Implements the abstract `Cost` class.

    This function computes the cross-entropy loss and its derivative, which is commonly used for classification tasks.

    Methods:
        value(self, y: np.ndarray, yp: np.ndarray) -> float:
            Computes the cross-entropy loss (cost) between true labels and predicted probabilities.

        derivative(self, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
            Computes the derivative (gradient) of the cross-entropy loss with respect to predicted probabilities.
    """
    def __init__(self):
        super().__init__()

    def value(self, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
      """
      Computes the cross-entropy loss between the true labels and predicted probabilities.

      Args:
          y (np.ndarray): True labels, usually one-hot encoded or a probability distribution (shape: (n_samples, n_classes)).
          yp (np.ndarray): Predicted probabilities (shape: (n_samples, n_classes)).
      Returns:
          float: The computed cross-entropy loss for the given true labels and predicted probabilities.
      Raises:
          ValueError: If `y` and `yp` shapes do not match.
      """
      if y.shape != yp.shape:
          raise ValueError(f"True labels and predicted probabilities must have the same shape. Got {y.shape} and {yp.shape}.")

      eps = 1e-15
      yp = np.clip(yp, eps, 1 - eps)

      loss = -np.mean(y * np.log(yp) + (1 - y) * np.log(1 - yp))
      return np.array([loss])


    def derivative(self, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """
        Computes the derivative (gradient) of the cross-entropy loss with respect to predicted probabilities.

        Args:
            y (np.ndarray): True labels, usually one-hot encoded or a probability distribution
                             (shape: (n_samples, n_classes)).
            yp (np.ndarray): Predicted probabilities (shape: (n_samples, n_classes)).

        Returns:
            np.ndarray: The gradient (derivative) of the cross-entropy loss with respect to the predicted probabilities.

        Raises:
            ValueError: If `y` and `yp` shapes do not match.
            ValueError: If any predicted probability is outside the range (0, 1).
        """
        if y.shape != yp.shape:
            raise ValueError(
                f"True labels and predicted probabilities must have the same shape. Got {y.shape} and {yp.shape}.")

        eps = 1e-15
        yp = np.clip(yp, eps, 1 - eps)

        return -(y / yp) + (1 - y) / (1 - yp)