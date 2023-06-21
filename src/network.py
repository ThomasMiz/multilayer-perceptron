import numpy as np
from src.theta import ThetaFunction


class Network:
    def __init__(self, input_size: int, arch: list[tuple[int, ThetaFunction]]) -> None:
        if len(arch) < 2:
            raise ValueError("A network must have at least two layers")

        self.input_size = input_size
        self.arch = arch

        # Stores the weights of the inputs for each layer. Each layer's weights are represented by a matrix
        # Each matrix's size is decided by sizes of it's respective layer and it's previous layer (or input, for the first layer)
        self.weights_per_layer = []
        for i in range(len(arch)):
            # Initialize all weights as random values between -1 and 1
            prev_layer_size = input_size if i == 0 else arch[i - 1][0]
            self.weights_per_layer.append(np.round(np.random.uniform(-1, 1, (prev_layer_size, arch[i][0])), 1))

    def evaluate(self, input: np.ndarray):
        input = np.array(input)
        if input.ndim != 1:
            raise ValueError("The input must have only 1 dimention")
        if len(input) != self.input_size:
            raise ValueError("The input size must match the network's input size")

        # Feedforward
        prev_layer_state = input
        for i in range(len(self.arch)):
            layer_weights = self.weights_per_layer[i]
            layer_theta = self.arch[i][1]
            h_matrix = np.matmul(prev_layer_state, layer_weights)
            prev_layer_state = layer_theta.primary(h_matrix)

        return prev_layer_state


class NetworkTrainer:
    def __init__(self, network: Network, eta: float) -> None:
        self.network = network
        self.eta = eta

    def evaluate_and_adjust(self, input: np.ndarray, expected_output: np.ndarray):
        input = np.array(input)
        if input.ndim != 1:
            raise ValueError("The input must have only 1 dimention")
        if len(input) != self.network.input_size:
            raise ValueError("The input size must match the network's input size")

        # Feedforward
        h_matrix_per_layer = [None]
        states_per_layer = [input]
        for i in range(len(self.network.arch)):
            layer_weights = self.network.weights_per_layer[i]
            layer_theta = self.network.arch[i][1]
            h_matrix = np.matmul(states_per_layer[-1], layer_weights)
            h_matrix_per_layer.append(h_matrix)
            states_per_layer.append(layer_theta.primary(h_matrix))

        # Backpropagation
        ds_matrix_per_layer = [None] * len(self.network.arch)
        dw_matrix_per_layer = [None] * len(self.network.arch)

        # For last layer
        layer_theta = self.network.arch[-1][1]
        ds_matrix_per_layer[-1] = (expected_output - states_per_layer[-1]) * layer_theta.derivative(states_per_layer[-1], h_matrix_per_layer[-1])
        dw_matrix_per_layer[-1] = self.eta * ds_matrix_per_layer[-1] * states_per_layer[-2]
        if len(dw_matrix_per_layer[-1].shape) == 1:
            dw_matrix_per_layer[-1] = dw_matrix_per_layer[-1][:, None]

        # For inner layers
        for i in range(len(self.network.arch) - 2, -1, -1):
            layer_theta = self.network.arch[i][1]
            ds_matrix_per_layer[i] = np.matmul(ds_matrix_per_layer[i + 1], self.network.weights_per_layer[i + 1].T) * layer_theta.derivative(states_per_layer[i + 1], h_matrix_per_layer[i])
            dw_matrix_per_layer[i] = self.eta * ds_matrix_per_layer[i] * states_per_layer[i][:, None]
            if len(dw_matrix_per_layer[i].shape) == 1:
                dw_matrix_per_layer[i] = dw_matrix_per_layer[i][:, None]

        for i in range(len(self.network.weights_per_layer)):
            self.network.weights_per_layer[i] += dw_matrix_per_layer[i]

    def train(self, dataset: list[np.ndarray], expected_outputs: list[np.ndarray]):
        tutu = True
        while (tutu):
            for i in range(len(dataset)):
                self.evaluate_and_adjust(dataset[i], expected_outputs[i])

            tutu = False
            for i in range(len(dataset)):
                obtained = self.network.evaluate(dataset[i])
                expected = expected_outputs[i]
                if not np.array_equal(obtained, expected):
                    tutu = True
