import numpy as np
from abc import ABC, abstractmethod
from src.network import Network

class Optimizer(ABC):
    """
    Represents an optimizer for the backpropagation algorithm, which works by applying additional terms to the delta-weights matrix
    before the weights are updated with it.
    """

    def initialize(self, network: Network):
        """Optimizers may implement this method to perform initialization before training starts."""
        pass

    @abstractmethod
    def apply(self, layer_number: int, learning_date: float, dw: np.ndarray) -> np.ndarray:
        """Calculates the delta-weights matrix to apply for a layer and returns it."""
        pass


class GradientDescentOptimizer(Optimizer):
    """The simplest optimizer, which runs backpropagation without any additional terms."""
    def apply(self, layer_number: int, learning_date: float, dw: np.ndarray) -> np.ndarray:
        return learning_date * dw
