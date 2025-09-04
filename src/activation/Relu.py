import numpy as np
import Activation
from abc import ABC, abstractmethod

# Documentar los métodos implementados
class Relu(Activation):
  """ Función de activación RELU. Implementa Activación
  """
  def Relu(self):
    pass

  def value(self, X: np.ndarray) -> np.ndarray:
    pass

  def derivative(self, X: np.ndarray) -> np.ndarray:
    pass