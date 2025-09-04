import numpy as np
import Metric
from abc import ABC, abstractmethod

# Documentar los métodos implementados
class Accuracy(Metric):
  """ Metrica de exactitud (acertados / totales). Implementa Metric
  """
  def Acurracy(self):
    pass

  def value(self, Y: np.ndarray, Yp: np.ndarray) -> np.ndarray:
    pass