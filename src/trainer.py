import numpy as np
from src.network import Network
from src.error import ErrorFunction
from src.utils import columnarize, add_lists
from src.optimizer import Optimizer


class NetworkTrainer:
    """An object used for training a given neural network."""

    def __init__(self, network: Network, learning_rate: float, error_function: ErrorFunction, optimizer: Optimizer) -> None:
        self.network = network
        self.learning_rate = learning_rate
        self.error_function = error_function
        self.optimizer = optimizer

        self.optimizer.initialize(network)

    def evaluate_and_adjust(self, input: np.ndarray, expected_output: np.ndarray, epoch_number: int) -> list[np.ndarray]:
        if input.ndim != 1:
            raise ValueError("The input must have only 1 dimention")
        if len(input) != self.network.input_size:
            raise ValueError("The input size must match the network's input size")

        # Feedforward
        h_vector_per_layer = [None]
        outputs_per_layer = [input]
        for i in range(self.network.layer_count):
            weights = self.network.layer_weights[i]
            activation = self.network.layer_activations[i]
            h_vector = np.matmul(outputs_per_layer[-1], weights[1:])
            np.add(h_vector, weights[0], out=h_vector)
            h_vector_per_layer.append(h_vector)
            outputs_per_layer.append(activation.primary(h_vector))

        # Backpropagation
        s_vector_per_layer = [None] * self.network.layer_count

        # For last layer
        activation = self.network.layer_activations[-1]
        s_vector_per_layer[-1] = (expected_output - outputs_per_layer[-1]) * activation.derivative(outputs_per_layer[-1], h_vector_per_layer[-1])

        # For inner layers
        for i in range(self.network.layer_count - 2, -1, -1):
            weights = self.network.layer_weights[i + 1]
            activation = self.network.layer_activations[i]
            s_vector_per_layer[i] = np.matmul(s_vector_per_layer[i + 1], weights[1:].T) * activation.derivative(outputs_per_layer[i + 1], h_vector_per_layer[i + 1])

        # Calculate delta weights matrices
        dw_matrix_per_layer = [None] * self.network.layer_count
        for i in range(self.network.layer_count):
            dw_matrix_per_layer[i] = columnarize(self.optimizer.apply(i, self.learning_rate, np.concatenate((np.ones(1), outputs_per_layer[i]))[:, None] * s_vector_per_layer[i]))

        return dw_matrix_per_layer

    def train(self, dataset: list[np.ndarray], expected_outputs: list[np.ndarray], acceptable_error):
        epoch = 0
        while (True):
            epoch += 1
            self.optimizer.start_next_epoch(epoch)
            weights_adjustments = [np.zeros_like(w) for w in self.network.layer_weights]
            for i in range(len(dataset)):
                add_lists(weights_adjustments, self.evaluate_and_adjust(dataset[i], expected_outputs[i], epoch))
            self.network.adjust_weights(weights_adjustments)

            outputs = [self.network.evaluate(d) for d in dataset]
            err = self.error_function.error_for_dataset(np.array(expected_outputs), np.array(outputs))
            print(f"Error after epoch {epoch}: {err}")
            if err <= acceptable_error:
                break

    def train_analyze(self, dataset: list[np.ndarray], expected_outputs: list[np.ndarray], acceptable_error, max_epochs=None):
        epoch = 0
        error_history = []
        error_history.append(self.error_function.error_for_dataset(np.array(expected_outputs), np.array([self.network.evaluate(d) for d in dataset])))
        finish_reason = 'unspecified'
        while (True):
            epoch += 1
            self.optimizer.start_next_epoch(epoch)
            weights_adjustments = [np.zeros_like(w) for w in self.network.layer_weights]
            for i in range(len(dataset)):
                add_lists(weights_adjustments, self.evaluate_and_adjust(dataset[i], expected_outputs[i], epoch))
            self.network.adjust_weights(weights_adjustments)

            outputs = [self.network.evaluate(d) for d in dataset]
            err = self.error_function.error_for_dataset(np.array(expected_outputs), np.array(outputs))
            error_history.append(err)
            if err <= acceptable_error:
                finish_reason = 'Acceptable error reached'
                break
            if max_epochs is not None and epoch >= max_epochs:
                finish_reason = 'Max epochs reached'
                break
        return epoch, finish_reason, error_history
