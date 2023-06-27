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

    def __init__(self, config=None) -> None:
        pass

    def primary(self, x: np.ndarray) -> np.ndarray:
        return (x >= 0) * 2 - 1

    def derivative(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.ones_like(p)


class LinealActivationFunction(ActivationFunction):
    """An identity function, returns the value unmodified."""

    def __init__(self, config=None) -> None:
        pass

    def primary(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, p: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.ones_like(p)


class TanhActivationFunction(ActivationFunction):
    """An activation function whose image is (-1, 1)."""
    range = (-1, 1)

    def __init__(self, config) -> None:
        if 'beta' not in config or config['beta'] is None:
            raise Exception('TanhActivationFunction requires a beta parameter')
        self.beta = float(config['beta'])

    def primary(self, x: float) -> float:
        return np.tanh(self.beta * x)

    def derivative(self, p: float, x: float) -> float:
        return self.beta * (1 - p*p)


class LogisticActivationFunction(ActivationFunction):
    """A logistic function whose image is (0, 1)."""
    range = (0, 1)

    def __init__(self, config) -> None:
        if 'beta' not in config or config['beta'] is None:
            raise Exception('LogisticActivationFunction requires a beta parameter')
        self.beta = float(config['beta'])

    def primary(self, x: float) -> float:
        return 1 / (1 + np.exp(-2 * self.beta * x))

    def derivative(self, p: float, x: float) -> float:
        return 2 * self.beta * p * (1 - p)


map = {
    "simple": SimpleActivationFunction,
    "lineal": LinealActivationFunction,
    "tanh": TanhActivationFunction,
    "logistic": LogisticActivationFunction
}
