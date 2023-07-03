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

    def __init__(self, config=None) -> None:
        pass

    def apply(self, layer_number: int, learning_date: float, dw: np.ndarray) -> np.ndarray:
        return learning_date * dw


class MomentumOptimizer(Optimizer):
    """A simple optimizer that adds a percentage of the previous iteration's delta_w to make a momentum effect."""

    def __init__(self, config) -> None:
        if 'alpha' not in config or config['alpha'] is None:
            raise Exception('MomentumOptimizer requires an alpha parameter')
        self.alpha = float(config['alpha'])
        if self.alpha <= 0 or self.alpha >= 1:
            print(f"⚠️⚠️⚠️ Warning: MomentumOptimizer received alpha outside of range (0, 1): {self.alpha}")

    def initialize(self, network: Network):
        self.previous_dw = [np.zeros_like(weights) for weights in network.layer_weights]

    def apply(self, layer_number: int, learning_date: float, dw: np.ndarray) -> np.ndarray:
        dw_matrix =  learning_date * dw + self.alpha * self.previous_dw[layer_number]
        self.previous_dw[layer_number] = dw_matrix
        return dw_matrix