import numpy as np
import json
from src.activation import ActivationFunction, activation_function_from_json
from src.utils import add_lists, ndarray_list_from_json, ndarray_list_to_json
from src.weights_initializer import WeightsInitializer


class Network:
    """Represents a multilayer perceptron."""

    def __init__(self, input_size: int, arch: list[tuple[int, ActivationFunction]], weight_initializer: (WeightsInitializer | list[np.ndarray])) -> None:
        """
        Creates a new Network with the specified architecture, represented as a list of tuples where [0] is the amount of neurons and [1] is the
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

        # Initialization may be done by specifying weight_initializer to a list or a WeightsInitializer. 
        if isinstance(weight_initializer, list):
            if len(weight_initializer) != self.layer_count:
                raise ValueError('Failed to initialize network: amount of weight matrices does not match amount of layers')
            prev_layer_size = input_size
            for i in range(self.layer_count):
                if (prev_layer_size + 1, self.layer_sizes[i]) != weight_initializer[i].shape:
                    raise ValueError('Failed to initialize network: weights matrices do not match network architecture')
                prev_layer_size = self.layer_sizes[i]
            self.layer_weights = weight_initializer
        else:
            prev_layer_size = input_size
            for i in range(self.layer_count):
                # Initialize weights and biases
                weights_and_biases = weight_initializer.get_weights(i, self.layer_sizes[i], prev_layer_size)
                self.layer_weights.append(np.vstack(weights_and_biases))
                prev_layer_size = self.layer_sizes[i]

    @property
    def output_size(self):
        return self.layer_sizes[-1]

    def evaluate_with_storage(self, input: np.ndarray, h_vectors_out: list[np.ndarray], state_vectors_out: list[np.ndarray]) -> np.ndarray:
        """Calculates this network's output vector for a given input vector, storing results in the provided numpy vectors. Skips checks."""
        prev_layer_output = input
        for i in range(self.layer_count):
            h_vector = np.matmul(prev_layer_output, self.layer_weights[i][1:], out=h_vectors_out[i])
            np.add(h_vector, self.layer_weights[i][0], out=h_vector)
            prev_layer_output = self.layer_activations[i].primary(h_vector, out=state_vectors_out[i])

        return prev_layer_output

    def evaluate(self, input: np.ndarray) -> np.ndarray:
        """Calculates this network's output vector for a given input vector."""
        if input.ndim != 1:
            raise ValueError("The input must have only 1 dimention")
        if len(input) != self.input_size:
            raise ValueError("The input size must match the network's input size")

        # Feedforward
        prev_layer_output = input
        for i in range(self.layer_count):
            # layer_output = activation.primary(np.matmul(prev_layer_output, layer_weights)) + layer_biases
            h_vector = np.matmul(prev_layer_output, self.layer_weights[i][1:])
            np.add(h_vector, self.layer_weights[i][0], out=h_vector)
            prev_layer_output = self.layer_activations[i].primary(h_vector, out=h_vector)

        return prev_layer_output

    def adjust_weights(self, dw_matrix_per_layer: list[np.ndarray]) -> None:
        add_lists(self.layer_weights, dw_matrix_per_layer)

    def to_json(self):
        return {
            "architecture": [{"size": self.layer_sizes[i], "activation": self.layer_activations[i].to_json()} for i in range(self.layer_count)],
            "layer_weights": ndarray_list_to_json(self.layer_weights)
        }

    def save_to_file(self, file: str, indent: bool=False):
        with open(file, 'w') as f:
            json.dump(self.to_json(), f, indent=(4 if indent else None))

    def from_json(d: dict):
        architecture = [(int(x["size"]), activation_function_from_json(x["activation"])) for x in d["architecture"]]
        layer_weights = ndarray_list_from_json(d["layer_weights"])
        input_size = layer_weights[0].shape[0] - 1
        return Network(input_size=input_size, arch=architecture, weight_initializer=layer_weights)

    def __repr__(self) -> str:
        return f"Network: {self.input_size} inputs, {self.layer_count} layers sizes {self.layer_sizes}"

    def __str__(self) -> str:
        return self.__repr__()
