import numpy as np
from abc import ABC, abstractmethod


class ThetaFunction(ABC):
    @abstractmethod
    def primary(self, x: float) -> float:
        """Used to transform the output of a perceptron."""
        pass

    @abstractmethod
    def derivative(self, p: float, x: float) -> float:
        """Used to multiply the delta_w of a perceptron while training."""
        pass


class SimpleThetaFunction(ThetaFunction):
    """A theta function that returns 1 if x >= 0, -1 otherwise."""
    range = (-1, 1)

    def __init__(self, config=None) -> None:
        pass

    def primary(self, x: float) -> float:
        return (x >= 0) * 2 - 1

    def derivative(self, p: float, x: float) -> float:
        return np.ones_like(p)


class LinealThetaFunction(ThetaFunction):
    """An identity function, returns the value unmodified."""

    def __init__(self, config=None) -> None:
        pass

    def primary(self, x: float) -> float:
        return x

    def derivative(self, p: float, x: float) -> float:
        return 1


class TanhThetaFunction(ThetaFunction):
    """A theta function whose image is (-1, 1)."""
    range = (-1, 1)

    def __init__(self, config) -> None:
        if 'beta' not in config or config['beta'] is None:
            raise Exception('TanhThetaFunction requires a beta parameter')
        self.beta = float(config['beta'])

    def primary(self, x: float) -> float:
        return np.tanh(self.beta * x)

    def derivative(self, p: float, x: float) -> float:
        return self.beta * (1 - p*p)


class LogisticThetaFunction(ThetaFunction):
    """A logistic function whose image is (0, 1)."""
    range = (0, 1)

    def __init__(self, config) -> None:
        if 'beta' not in config or config['beta'] is None:
            raise Exception('LogisticThetaFunction requires a beta parameter')
        self.beta = float(config['beta'])

    def primary(self, x: float) -> float:
        return 1 / (1 + np.exp(-2 * self.beta * x))

    def derivative(self, p: float, x: float) -> float:
        return 2 * self.beta * p * (1 - p)


map = {
    "simple": SimpleThetaFunction,
    "lineal": LinealThetaFunction,
    "tanh": TanhThetaFunction,
    "logistic": LogisticThetaFunction
}
