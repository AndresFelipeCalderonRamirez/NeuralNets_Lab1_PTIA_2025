from .Activation import Activation
from .Relu import Relu
from .Sigmoid import Sigmoid

Activation.activation_map['relu'] = Relu
Activation.activation_map['sigmoid'] = Sigmoid