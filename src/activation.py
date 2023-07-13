import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """Represents a neuron activation function. The methods of this class must be able to operate in vector form."""

    @abstractmethod
    def primary(self, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        """Used to transform the output of a perceptron."""
        pass

    @abstractmethod
    def derivative(self, p: np.ndarray, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        """Used to multiply the delta_w of a perceptron while training. Receives p, the value of primary(x), and x."""
        pass

    @abstractmethod
    def to_json(self) -> dict:
        """Serializes this ActivationFunction to a dict."""
        pass


class SimpleActivationFunction(ActivationFunction):
    """An activation function that returns 1 if x >= 0, -1 otherwise."""
    range = (-1, 1)

    def __init__(self) -> None:
        pass

    def primary(self, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        # (x >= 0) * 2 - 1
        if out is None:
            out = np.zeros_like(x)
        np.greater_equal(x, 0, out=out)
        np.multiply(out, 2, out=out)
        return np.subtract(out, 1, out=out)

    def derivative(self, p: np.ndarray, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        if out is None:
            return np.ones_like(p)
        out.fill(1)
        return out

    def to_json(self) -> dict:
        return {"type": "simple"}

    def __repr__(self) -> str:
        return "SimpleActivationFunction"

    def __str__(self) -> str:
        return self.__repr__()


class IdentityActivationFunction(ActivationFunction):
    """An identity function, returns the value unmodified."""

    def __init__(self) -> None:
        pass

    def primary(self, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        return x

    def derivative(self, p: np.ndarray, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        if out is None:
            return np.ones_like(p)
        out.fill(1)
        return out

    def to_json(self) -> dict:
        return {"type": "identity"}

    def __repr__(self) -> str:
        return "IdentityActivationFunction"

    def __str__(self) -> str:
        return self.__repr__()


class ReluActivationFunction(ActivationFunction):
    """An activation function that returns max(x, 0)."""

    def __init__(self) -> None:
        pass

    def primary(self, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        return np.maximum(x, 0, out=out)

    def derivative(self, p: np.ndarray, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        # (x >= 0) * 1
        if out is None:
            out = np.zeros_like(x)
        return np.greater_equal(x, 0, out=out)

    def to_json(self) -> dict:
        return {"type": "relu"}

    def __repr__(self) -> str:
        return "ReluActivationFunction"

    def __str__(self) -> str:
        return self.__repr__()


class TanhActivationFunction(ActivationFunction):
    """An activation function whose image is (-1, 1)."""
    range = (-1, 1)

    def __init__(self, beta: float=1.0) -> None:
        self.beta = beta

    def primary(self, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        # np.tanh(self.beta * x)
        out = np.multiply(x, self.beta, out=out)
        return np.tanh(out, out=out)

    def derivative(self, p: np.ndarray, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        # self.beta * (1 - p*p)
        out = np.square(p, out=out)
        np.subtract(1, out, out=out)
        return np.multiply(out, self.beta, out=out)

    def to_json(self) -> dict:
        return {"type": "tanh", "beta": self.beta}

    def __repr__(self) -> str:
        return f"TanhActivationFunction beta={self.beta}"

    def __str__(self) -> str:
        return self.__repr__()


class LogisticActivationFunction(ActivationFunction):
    """A logistic function whose image is (0, 1)."""
    range = (0, 1)

    def __init__(self, beta: float=0.5) -> None:
        self.minus_beta_times_two = -2 * beta

    def primary(self, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        # 1 / (1 + np.exp(self.minus_beta_times_two * x))
        out = np.multiply(x, self.minus_beta_times_two, out=out)
        np.exp(out, out=out)
        np.add(out, 1, out=out)
        return np.divide(1, out, out=out)

    def derivative(self, p: np.ndarray, x: np.ndarray, out: np.ndarray=None) -> np.ndarray:
        # self.minus_beta_times_two * p * (p - 1)
        out = np.subtract(p, 1, out=out)
        np.multiply(out, p, out=out)
        return np.multiply(out, self.minus_beta_times_two, out=out)

    def to_json(self) -> dict:
        return {"type": "logistic", "beta": self.minus_beta_times_two / -2}

    def __repr__(self) -> str:
        return f"TanhActivationFunction beta={self.minus_beta_times_two / -2}"

    def __str__(self) -> str:
        return self.__repr__()


activation_function_map = {
    "simple": SimpleActivationFunction,
    "indentity": IdentityActivationFunction,
    "relu": ReluActivationFunction,
    "tanh": TanhActivationFunction,
    "logistic": LogisticActivationFunction
}


def activation_function_from_json(d: dict) -> ActivationFunction:
    class_type = activation_function_map[d["type"]]
    if class_type is TanhActivationFunction or class_type is LogisticActivationFunction:
        return class_type(beta=float(d["beta"]))
    return class_type()
