import numpy as np
from src.activation import ActivationFunction
from src.utils import columnarize, add_lists


class Network:
    """Represents a multilayer perceptron."""

    def __init__(self, input_size: int, arch: list[tuple[int, ActivationFunction]]) -> None:
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
        layer (or input vector) and the neurons of the current layer.
        """

        for i in range(self.layer_count):
            # Initialize all weights as random values between -1 and 1
            prev_layer_size = input_size if i == 0 else self.layer_sizes[i - 1]
            self.layer_weights.append(np.round(np.random.uniform(-1, 1, (prev_layer_size + 1, self.layer_sizes[i])), 1))

    def evaluate(self, input: np.ndarray):
        """Calculates this network's output vector for a given input vector."""
        if input.ndim != 1:
            raise ValueError("The input must have only 1 dimention")
        if len(input) != self.input_size:
            raise ValueError("The input size must match the network's input size")

        # Feedforward
        prev_layer_output = input
        for i in range(self.layer_count):
            # We prepend the input with a 1 to facilitate matrix multiplication, since the first row of the matrix are the biases.
            h_vector = np.matmul(np.concatenate((np.ones(1), prev_layer_output)), self.layer_weights[i])
            prev_layer_output = self.layer_activations[i].primary(h_vector)

        return prev_layer_output


class NetworkTrainer:
    """An object used for training a given neural network."""

    def __init__(self, network: Network, learning_rate: float) -> None:
        self.network = network
        self.learning_rate = learning_rate

    def evaluate_and_adjust(self, input: np.ndarray, expected_output: np.ndarray) -> list[np.ndarray]:
        if input.ndim != 1:
            raise ValueError("The input must have only 1 dimention")
        if len(input) != self.network.input_size:
            raise ValueError("The input size must match the network's input size")

        # Feedforward
        h_vector_per_layer = [None]
        outputs_per_layer = [input]
        for i in range(self.network.layer_count):
            layer_weights = self.network.layer_weights[i]
            layer_activation = self.network.layer_activations[i]
            h_vector = np.matmul(np.concatenate((np.ones(1), outputs_per_layer[-1])), layer_weights)
            h_vector_per_layer.append(h_vector)
            outputs_per_layer.append(layer_activation.primary(h_vector))

        # Backpropagation
        s_vector_per_layer = [None] * self.network.layer_count
        dw_matrix_per_layer = [None] * self.network.layer_count

        # For last layer
        layer_activation = self.network.layer_activations[-1]
        s_vector_per_layer[-1] = (expected_output - outputs_per_layer[-1]) * layer_activation.derivative(outputs_per_layer[-1], h_vector_per_layer[-1])
        dw_matrix_per_layer[-1] = columnarize(self.learning_rate * np.concatenate((np.ones(1), outputs_per_layer[-2])) * s_vector_per_layer[-1])

        # For inner layers
        for i in range(self.network.layer_count - 2, -1, -1):
            layer_activation = self.network.layer_activations[i]
            s_vector_per_layer[i] = np.matmul(s_vector_per_layer[i + 1], self.network.layer_weights[i + 1][1:].T) * layer_activation.derivative(outputs_per_layer[i + 1], h_vector_per_layer[i + 1])
            dw_matrix_per_layer[i] = columnarize(self.learning_rate * np.concatenate((np.ones(1), outputs_per_layer[i]))[:, None] * s_vector_per_layer[i])

        return dw_matrix_per_layer

    def adjust_weights(self, dw_matrix_per_layer: list[np.ndarray]) -> None:
        add_lists(self.network.layer_weights, dw_matrix_per_layer)

    def train(self, dataset: list[np.ndarray], expected_outputs: list[np.ndarray]):
        tutu = True
        while (tutu):
            weights_adjustments = [np.zeros_like(w) for w in self.network.layer_weights]
            for i in range(len(dataset)):
                add_lists(weights_adjustments, self.evaluate_and_adjust(dataset[i], expected_outputs[i]))
            self.adjust_weights(weights_adjustments)

            tutu = False
            for i in range(len(dataset)):
                obtained = self.network.evaluate(dataset[i])
                expected = expected_outputs[i]
                if not np.array_equal(obtained, expected):
                    tutu = True
