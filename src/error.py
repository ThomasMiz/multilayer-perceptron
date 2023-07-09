import numpy as np
from abc import ABC, abstractmethod

class ErrorFunction(ABC):
    """
    Represents a type of error function.
    The meaning of the returned values are up to interpretation for each type of error function, however the goal is always getting the lowest
    error value possible.
    """

    @abstractmethod
    def error_for_single(self, expected_output: np.ndarray, output: np.ndarray) -> float:
        """Calculates the error for a single output vector from the expected output and actual output."""
        pass

    @abstractmethod
    def error_for_dataset(self, expected_outputs: np.ndarray, outputs: np.ndarray) -> float:
        """Calculates the error for a set of output vectors."""
        pass


class CountNonmatchingErrorFunction(ErrorFunction):
    """An error function that counts how many output vectors do not exactly match their respective expected output vector."""

    def error_for_single(self, expected_output: np.ndarray, output: np.ndarray) -> float:
        return 1 - np.array_equal(expected_output, output)

    def error_for_dataset(self, expected_outputs: np.ndarray, outputs: np.ndarray) -> float:
        return len(expected_outputs) - np.sum(np.all(np.equal(expected_outputs, outputs), axis=1))


class CostAverageErrorFunction(ErrorFunction):
    """An error function that """

    def error_for_single(self, expected_output: np.ndarray, output: np.ndarray) -> float:
        tmp = np.subtract(expected_output, output)
        np.power(tmp, 2, out=tmp)
        return np.average(tmp) * 0.5

    def error_for_dataset(self, expected_outputs: np.ndarray, outputs: np.ndarray) -> float:
        tmp = (expected_outputs - outputs).sum(axis=1)
        np.power(tmp, 2, out=tmp)
        return np.average(tmp) * 0.5


map = {
    'count_nonmatching': CountNonmatchingErrorFunction,
    'cost_average': CostAverageErrorFunction
}
