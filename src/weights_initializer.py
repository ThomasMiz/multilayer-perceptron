import numpy as np
from abc import ABC, abstractmethod


class WeightsInitializer(ABC):
    """Represents a method of weight initialization."""

    @abstractmethod
    def get_weights(self, layer_number: int, layer_size: int, prev_layer_size: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple with the biases and weights (in that order) for a given layer. The biases are represented as a row vector
        while the weights are represented as an NxM matrix, where N is the number of neurons on the previous layer (or input),
        and M is the amount of neurons on the current layer.
        """
        pass


class RandomWeightsInitializer(WeightsInitializer):
    """Initializes weights and biases with uniform random numbers from a range."""

    def __init__(self, range=None, weights_range=None, biases_range=None, round_decimal_digits_to=None) -> None:
        """
        Creates a RandomWeightsInitializer. range=(min, max) may be used to set the range of both weights and biases, or biases_range=(min, max) and
        weights_range=(min, max) may be used separately. to specify different ranges for each. round_decimal_digits_to=n may be used to round the
        random values to n decimal digits (otherwise disabled by default).
        """

        self.weights_range = weights_range if weights_range is not None else (range if range is not None else (-1, 1))
        self.biases_range = biases_range if biases_range is not None else (range if range is not None else (-1, 1))
        self.round_decimal_digits_to = round_decimal_digits_to

    def get_weights(self, layer_number: int, layer_size: int, prev_layer_size: int) -> tuple[np.ndarray, np.ndarray]:
        biases = np.random.uniform(self.biases_range[0], self.biases_range[1], layer_size)
        weights = np.random.uniform(self.weights_range[0], self.weights_range[1], (prev_layer_size, layer_size))
        if self.round_decimal_digits_to is not None:
            biases = np.round(biases, self.round_decimal_digits_to)
            weights = np.round(weights, self.round_decimal_digits_to)

        return biases, weights


class ValuesWeightsInitializer(WeightsInitializer):
    """Initializes weights and biases with predetermined values."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """
        Creates a ValuesWeightsInitializer. Receives a list of weights matrices and a list of bias vectors, one per layer, ordered by layer.
        """

        if weights is None or len(weights) == 0:
            raise ValueError('weights may not be null nor empty')
        if biases is None or len(biases) == 0:
            raise ValueError('biases may not be null nor empty')
        if len(weights) != len(biases):
            raise ValueError('The length of the weights and biases lists must be the same')

        self.weights = weights
        self.biases = biases

    def get_weights(self, layer_number: int, layer_size: int, prev_layer_size: int) -> tuple[np.ndarray, np.ndarray]:
        if self.biases[layer_number].shape != (layer_size,):
            raise ValueError(f"The shape of the bias vector for layer {layer_number} {self.biases[layer_number].shape} doesn\'t match what the network expected {(layer_size,)}")
        if self.weights[layer_number].shape != (prev_layer_size, layer_size):
            raise ValueError(f"The shape of the weights matrix for layer {layer_number} {self.biases[layer_number].shape} doesn\'t match what the network expected {(prev_layer_size, layer_size)}")

        # The user may have specified matrices with integers, this ensures they are properly converted so all operations are float64.
        return np.asarray(self.biases[layer_number], dtype='float64'), np.asarray(self.weights[layer_number], dtype='float64')
