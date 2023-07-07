import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """Represents a neuron activation function. The methods of this class must be able to operate in vector form."""

    @abstractmethod
    def primary(self, x: np.ndarray) -> np.ndarray:
        """Used to transform the output of a perceptron."""
        pass

    @abstractmethod
    def derivative(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Used to multiply the delta_w of a perceptron while training."""
        pass


class SimpleActivationFunction(ActivationFunction):
    """An activation function that returns 1 if x >= 0, -1 otherwise."""
    range = (-1, 1)

    def __init__(self) -> None:
        pass

    def primary(self, x: np.ndarray) -> np.ndarray:
        return (x >= 0) * 2 - 1

    def derivative(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.ones_like(p)


class IdentityActivationFunction(ActivationFunction):
    """An identity function, returns the value unmodified."""

    def __init__(self) -> None:
        pass

    def primary(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.ones_like(p)


class ReluActivationFunction(ActivationFunction):
    """An activation function that returns max(x, 0)."""

    def __init__(self) -> None:
        pass

    def primary(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def derivative(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return (x >= 0) * 1


class TanhActivationFunction(ActivationFunction):
    """An activation function whose image is (-1, 1)."""
    range = (-1, 1)

    def __init__(self, beta: float=1.0) -> None:
        self.beta = beta

    def primary(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * x)

    def derivative(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.beta * (1 - p*p)


class LogisticActivationFunction(ActivationFunction):
    """A logistic function whose image is (0, 1)."""
    range = (0, 1)

    def __init__(self, beta: float=0.5) -> None:
        self.minus_beta_times_two = -2 * beta

    def primary(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(self.minus_beta_times_two * x))

    def derivative(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.minus_beta_times_two * p * (p - 1)


map = {
    "simple": SimpleActivationFunction,
    "indentity": IdentityActivationFunction,
    "relu": ReluActivationFunction,
    "tanh": TanhActivationFunction,
    "logistic": LogisticActivationFunction
}
