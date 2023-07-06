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
    def apply(self, layer_number: int, learning_date: float, epoch_number: int, dw: np.ndarray) -> np.ndarray:
        """Calculates the delta-weights matrix to apply for a layer and returns it."""
        pass


class GradientDescentOptimizer(Optimizer):
    """The simplest optimizer, which runs backpropagation without any additional terms."""

    def __init__(self) -> None:
        pass

    def parse(config=None):
        pass

    def apply(self, layer_number: int, learning_date: float, epoch_number: int, dw: np.ndarray) -> np.ndarray:
        return learning_date * dw


class MomentumOptimizer(Optimizer):
    """A simple optimizer that adds a percentage of the previous iteration's delta_w to make a momentum effect."""

    def __init__(self, alpha: float=0.8) -> None:
        self.alpha = alpha
        if self.alpha <= 0 or self.alpha >= 1:
            print(f"⚠️⚠️⚠️ Warning: MomentumOptimizer received alpha outside of range (0, 1): {self.alpha}")

    def initialize(self, network: Network):
        self.previous_dw = [np.zeros_like(weights) for weights in network.layer_weights]

    def apply(self, layer_number: int, learning_date: float, epoch_number: int, dw: np.ndarray) -> np.ndarray:
        dw_matrix =  learning_date * dw + self.alpha * self.previous_dw[layer_number]
        self.previous_dw[layer_number] = dw_matrix
        return dw_matrix


class RMSPropOptimizer(Optimizer):
    """An optimizer based on root mean squares."""

    def __init__(self, gamma: float=0.9, epsilon: float=1e-8) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        if self.gamma <= 0 or self.gamma >= 1:
            print(f"⚠️⚠️⚠️ Warning: RMSPropOptimizer received alpha outside of range (0, 1): {self.gamma}")
        if self.epsilon <= 0 or self.epsilon >= 1:
            print(f"⚠️⚠️⚠️ Warning: RMSPropOptimizer received negative or large epsilon: {self.epsilon}")

    def initialize(self, network: Network):
        self.previous_s_matrix = [np.zeros_like(weights) for weights in network.layer_weights]

    def apply(self, layer_number: int, learning_date: float, epoch_number: int, dw: np.ndarray) -> np.ndarray:
        s_matrix =  self.gamma * self.previous_s_matrix[layer_number] + (1 - self.gamma) * np.square(dw)
        self.previous_s_matrix[layer_number] = s_matrix
        return learning_date / np.sqrt(s_matrix + self.epsilon) * dw


class AdamOptimizer(Optimizer):
    """An optimizer that combines the concepts of RMSProp and Momentum."""

    def __init__(self, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        if self.beta1 < 0 or self.beta1 >= 1:
            print(f"⚠️⚠️⚠️ Warning: AdamOptimizer received beta1 outside of range [0, 1): {self.beta1}")
        if self.beta2 < 0 or self.beta2 >= 1:
            print(f"⚠️⚠️⚠️ Warning: AdamOptimizer received beta2 outside of range [0, 1): {self.beta2}")
        if self.epsilon <= 0 or self.epsilon >= 1:
            print(f"⚠️⚠️⚠️ Warning: AdamOptimizer received negative or large epsilon: {self.epsilon}")

    def initialize(self, network: Network):
        self.m_per_layer = [np.zeros_like(weights) for weights in network.layer_weights]
        self.v_per_layer = [np.zeros_like(weights) for weights in network.layer_weights]

    def apply(self, layer_number: int, learning_date: float, epoch_number: int, dw: np.ndarray) -> np.ndarray:
        self.m_per_layer[layer_number] = self.beta1 * self.m_per_layer[layer_number] + (1 - self.beta1) * dw
        self.v_per_layer[layer_number] = self.beta2 * self.v_per_layer[layer_number] + (1 - self.beta2) * np.square(dw)

        m_hat = self.m_per_layer[layer_number] / (1 - np.power(self.beta1, epoch_number))
        v_hat = self.v_per_layer[layer_number] / (1 - np.power(self.beta2, epoch_number))

        dw_matrix = learning_date * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return dw_matrix
