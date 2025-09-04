import numpy as np
from ..activation import Activation
from abc import ABC, abstractmethod

class DenseLayer:
  """ Representa una capa (oculta o salida) en la red neuronal
  """
  def __init__(self, input_size: int, neurons: int, activation: str, learning_rate: float):
    """ inicializar una capa de neuronas dentro de la red neuronal.
    Args:
      input_size (int): número de neuronas de capa anterior o de atributos de entrada
      neurons (int): número de neuronas en la capa
      activation (str): nombre de la función de activación
      learning_rate (float): eta tasa de aprendizaje de la red
    """
    self.learninig_rate
    self.W
    self.b
    self.dW
    self.db
    pass

  def forward(self, Ab: np.ndarray):
    """ Transmite la entrada a partir del acumulativo de señales (f_base) y el potencial eléctrico (f_activación).
    Args:
      Ab (np.ndarray): características ó valores de activación de la capa anterior
    Return:
      S (np.ndarray): valores de activación de neuronas
    """
    pass

  def backward(self, Ab: np.ndarray):
    """ Transmite hacia atras el cambio del grandiente y el error (delta)
    Args:
      Ab (np.ndarray): características ó valores de error de la capa siguiente
      [capa salida] valores etiquetados esperados | [capa oculta] delta capa siguiente
    Return:
      S (np.ndarray): valor delta considerando gradiente y error
    """
    pass

  def update_parameters(self):
    """ Actualiza los parámetros de la capa a partir del gradiente y el error.
    Return:
    """
    pass

  def shapes(self):
    """ genera los valores asociados al tamaño de la capa
    Return:
      s (tupla<int>): tamaño de la capa
    """
    pass

  def to_string(self, detail: bool):
    """
    """
    pass