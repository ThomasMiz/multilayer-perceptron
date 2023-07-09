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

    def start_next_epoch(self, epoch_number: int):
        """Optimizers may implement this method to perform calculations at the beginning of each epoch."""
        pass

    @abstractmethod
    def apply(self, layer_number: int, learning_rate: float, dw: np.ndarray) -> np.ndarray:
        """Calculates the delta-weights matrix to apply for a layer and returns it."""
        pass


class GradientDescentOptimizer(Optimizer):
    """The simplest optimizer, which runs backpropagation without any additional terms."""

    def __init__(self) -> None:
        pass

    def parse(config=None):
        pass

    def apply(self, layer_number: int, learning_rate: float, dw: np.ndarray) -> np.ndarray:
        # learning_rate * dw
        return np.multiply(dw, learning_rate, out=dw)


class MomentumOptimizer(Optimizer):
    """A simple optimizer that adds a percentage of the previous iteration's delta_w to make a momentum effect."""

    def __init__(self, alpha: float=0.8) -> None:
        self.alpha = alpha
        if self.alpha <= 0 or self.alpha >= 1:
            print(f"⚠️⚠️⚠️ Warning: MomentumOptimizer received alpha outside of range (0, 1): {self.alpha}")

    def initialize(self, network: Network):
        self.previous_dw = [np.zeros_like(weights) for weights in network.layer_weights]

    def apply(self, layer_number: int, learning_rate: float, dw: np.ndarray) -> np.ndarray:
        # learning_rate * dw + self.alpha * self.previous_dw[layer_number]
        previous = self.previous_dw[layer_number]
        np.multiply(previous, self.alpha, out=previous)
        np.multiply(dw, learning_rate, out=dw)
        np.add(dw, previous, out=dw)
        np.copyto(previous, dw)
        return dw


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
        self.tmp_matrices = [np.zeros_like(weights) for weights in network.layer_weights]

    def apply(self, layer_number: int, learning_rate: float, dw: np.ndarray) -> np.ndarray:
        # s_matrix = self.gamma * self.previous_s_matrix[layer_number] + (1 - self.gamma) * np.square(dw)
        tmp = self.tmp_matrices[layer_number]
        np.square(dw, out=tmp)
        np.multiply(tmp, 1 - self.gamma, out=tmp)
        previous = self.previous_s_matrix[layer_number]
        np.multiply(previous, self.gamma, out=previous)
        np.add(previous, tmp, out=previous)

        # learning_rate / np.sqrt(s_matrix + self.epsilon) * dw
        np.copyto(tmp, previous)
        np.add(tmp, self.epsilon, out=tmp)
        np.sqrt(tmp, out=tmp)
        np.divide(learning_rate, tmp, out=tmp)
        return np.multiply(tmp, dw, out=dw)


class AdamOptimizer(Optimizer):
    """An optimizer that combines the concepts of RMSProp and Momentum."""

    def __init__(self, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.beta1_power_epoch = 1
        self.beta2_power_epoch = 1
        if self.beta1 < 0 or self.beta1 >= 1:
            print(f"⚠️⚠️⚠️ Warning: AdamOptimizer received beta1 outside of range [0, 1): {self.beta1}")
        if self.beta2 < 0 or self.beta2 >= 1:
            print(f"⚠️⚠️⚠️ Warning: AdamOptimizer received beta2 outside of range [0, 1): {self.beta2}")
        if self.epsilon <= 0 or self.epsilon >= 1:
            print(f"⚠️⚠️⚠️ Warning: AdamOptimizer received negative or large epsilon: {self.epsilon}")

    def initialize(self, network: Network):
        self.m_per_layer = [np.zeros_like(weights) for weights in network.layer_weights]
        self.v_per_layer = [np.zeros_like(weights) for weights in network.layer_weights]
        self.tmp_matrices = [np.zeros_like(weights) for weights in network.layer_weights]

    def start_next_epoch(self, epoch_number: int):
        self.beta1_power_epoch = np.power(self.beta1, epoch_number)
        self.beta2_power_epoch = np.power(self.beta2, epoch_number)

    def apply(self, layer_number: int, learning_rate: float, dw: np.ndarray) -> np.ndarray:
        tmp = self.tmp_matrices[layer_number]

        # self.m_per_layer[layer_number] = self.beta1 * self.m_per_layer[layer_number] + (1 - self.beta1) * dw
        m = self.m_per_layer[layer_number]
        np.multiply(m, self.beta1, out=m)
        np.multiply(dw, 1 - self.beta1, out=tmp)
        np.add(m, tmp, out=m)

        # self.v_per_layer[layer_number] = self.beta2 * self.v_per_layer[layer_number] + (1 - self.beta2) * np.square(dw)
        v = self.v_per_layer[layer_number]
        np.multiply(v, self.beta2, out=v)
        np.square(dw, out=tmp)
        np.multiply(tmp, 1 - self.beta2, out=tmp)
        np.add(v, tmp, out=v)

        # m_hat = self.m_per_layer[layer_number] / (1 - np.power(self.beta1, epoch_number))
        np.divide(m, 1 - self.beta1_power_epoch, out=dw)
        # v_hat = self.v_per_layer[layer_number] / (1 - np.power(self.beta2, epoch_number))
        np.divide(v, 1 - self.beta2_power_epoch, out=tmp)

        # dw = learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        np.sqrt(tmp, out=tmp)
        np.add(tmp, self.epsilon, out=tmp)
        np.multiply(dw, learning_rate, out=dw)
        return np.divide(dw, tmp, out=dw)
