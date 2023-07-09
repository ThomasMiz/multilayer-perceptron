import numpy as np
from src.activation import ActivationFunction
from src.utils import add_lists
from src.weights_initializer import WeightsInitializer


class Network:
    """Represents a multilayer perceptron."""

    def __init__(self, input_size: int, arch: list[tuple[int, ActivationFunction]], weight_initializer: WeightsInitializer) -> None:
        """
        Creates a new Network with the specified architechture, represented as a list of tuples where [0] is the amount of neurons and [1] is the
        activation function. All the neurons within a same layer use the same activation function.The first element of the list is the first layer,
        the one that receives the input vector.
        """

        if len(arch) < 2:
            raise ValueError("A network must have at least two layers")

        self.input_size = input_size
        """The size of the input vector."""

        self.layer_count = len(arch)
        """The amount of layers this network has."""

        self.layer_sizes = []
        """The amount of neurons in each layer."""

        self.layer_activations = []
        """The activation function for each layer."""

        for layer_tuple in arch:
            self.layer_sizes.append(layer_tuple[0])
            self.layer_activations.append(layer_tuple[1])

        self.layer_weights = []
        """
        The weights (and biases) for each layer, represented as a matrix in which w[0] are the biases, and w[1:] are the weights between the previous
        layer (or input vector) and the neurons of the current layer. Each column in the matrix represents the bias and weights for one neuron.
        """

        prev_layer_size = input_size
        for i in range(self.layer_count):
            # Initialize weights and biases
            weights_and_biases = weight_initializer.get_weights(i, self.layer_sizes[i], prev_layer_size)
            self.layer_weights.append(np.vstack(weights_and_biases))
            prev_layer_size = self.layer_sizes[i]

    def evaluate(self, input: np.ndarray) -> np.ndarray:
        """Calculates this network's output vector for a given input vector."""
        if input.ndim != 1:
            raise ValueError("The input must have only 1 dimention")
        if len(input) != self.input_size:
            raise ValueError("The input size must match the network's input size")

        # Feedforward
        prev_layer_output = input
        for i in range(self.layer_count):
            # We prepend the input with a 1 to facilitate matrix multiplication, since the first row of the matrix are the biases.
            h_vector = np.matmul(prev_layer_output, self.layer_weights[i][1:])
            np.add(h_vector, self.layer_weights[i][0], out=h_vector)
            prev_layer_output = self.layer_activations[i].primary(h_vector)

        return prev_layer_output

    def adjust_weights(self, dw_matrix_per_layer: list[np.ndarray]) -> None:
        add_lists(self.layer_weights, dw_matrix_per_layer)
