import numpy as np
import Cost
from abc import ABC, abstractmethod

# Documentar los métodos implementados
class CrossEntropy(Cost):
  """ Función de costo Entropia Cruzada. Implementa Cost
  """
  def CrossEntropy(self):
    pass

  def value(self, Y: np.ndarray, Yp: np.ndarray) -> np.ndarray:
    pass

  def derivative(self, Y: np.ndarray, Yp: np.ndarray) -> np.ndarray:
    pass