import numpy as np
import Activation
from abc import ABC, abstractmethod

#Documentar los métodos implementados
class Sigmoid(Activation):
  """ Función de activación sigmoide. Implementa Activación
  """
  def Sigmoid(self):
    pass

  def value(self, X: np.ndarray) -> np.ndarray:
    pass

  def derivative(self, X: np.ndarray) -> np.ndarray:
    pass