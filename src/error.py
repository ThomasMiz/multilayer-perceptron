import numpy as np
from abc import ABC, abstractmethod

class ErrorFunction(ABC):
    """
    Represents a type of error function.
    The meaning of the returned values are up to interpretation for each type of error function, however the goal is always getting the lowest
    error value possible.
    """

    @abstractmethod
    def error_for_single(self, expected_output: np.ndarray, actual_output: np.ndarray) -> float:
        """Calculates the error for a single output vector from the expected output and actual output."""
        pass

    @abstractmethod
    def error_for_dataset(self, expected_outputs: list[np.ndarray], actual_outputs: list[np.ndarray]) -> float:
        """Calculates the error for a set of output vectors."""
        pass

    @abstractmethod
    def to_json(self) -> dict:
        """Serializes this ErrorFunction to a dict."""
        pass


class CountNonmatchingErrorFunction(ErrorFunction):
    """An error function that counts how many output vectors do not exactly match their respective expected output vector."""

    def error_for_single(self, expected_output: np.ndarray, actual_output: np.ndarray) -> float:
        return 1 - np.array_equal(expected_output, actual_output)

    def error_for_dataset(self, expected_outputs: list[np.ndarray], actual_outputs: list[np.ndarray]) -> float:
        return len(expected_outputs) - np.sum(np.all(np.equal(expected_outputs, actual_outputs), axis=1))

    def to_json(self) -> dict:
        return {"type": "count_nonmatching"}


class CostAverageErrorFunction(ErrorFunction):
    """An error function that calculates the average of squares of the difference between the expected result and the actual result."""

    def error_for_single(self, expected_output: np.ndarray, actual_output: np.ndarray) -> float:
        tmp = np.subtract(expected_output, actual_output)
        np.square(tmp, out=tmp)
        return np.average(tmp) * 0.5

    def error_for_dataset(self, expected_outputs: list[np.ndarray], actual_outputs: list[np.ndarray]) -> float:
        tmp = np.subtract(expected_outputs, actual_outputs).sum(axis=1)
        np.square(tmp, out=tmp)
        return np.average(tmp) * 0.5

    def to_json(self) -> dict:
        return {"type": "cost_average"}


error_function_map = {
    'count_nonmatching': CountNonmatchingErrorFunction,
    'cost_average': CostAverageErrorFunction
}


def error_function_from_json(d: dict) -> ErrorFunction:
    class_type = error_function_map[d["type"]]
    return class_type()
