import numpy as np
import json
from src.network import Network
from src.error import ErrorFunction, error_function_from_json
from src.optimizer import Optimizer, optimizer_from_json
from src.utils import add_lists, columnarize, ndarray_list_from_json, ndarray_list_to_json


class NetworkTrainer:
    """An object used for training a given neural network."""

    def __init__(self, network: Network, learning_rate: float, error_function: ErrorFunction, optimizer: Optimizer, max_epochs: (int | None)=1000,
                 acceptable_error: (float | None)=0, record_error_history: bool=False) -> None:
        self.network = network
        self.learning_rate = learning_rate
        self.error_function = error_function
        self.optimizer = optimizer
        self.max_epochs = max_epochs

        self.acceptable_error = acceptable_error
        """An error value deemed as acceptable, and that training should stop once this or better is reached."""

        self.record_error_history = record_error_history
        """Whether to remember the error values calculated at the end of each epoch."""

        self.error_history = None
        """The error history, if error history is being recorded. error_history[i] contains the error after epoch i."""

        self.testset_error_history = None
        """The error history for the testset, if error history is being recorded. testset_error_history[i] contains the testset error after epoch i."""

        self.current_weights = None
        """
        During training, the current weights are stored in the Network object. Otherwise, the current weights are stored in this variable
        and the network instead contains the best weights so far.
        """

        self.current_error = None
        """The error value calculated at the last epoch."""

        self.current_testset_error = None
        """The testset error value calculated at the last epoch."""

        self.best_error = None
        """The lower error value calculated for an epoch so far."""

        self.current_epoch = 0
        """The current epoch number, or 0 if training hasn't started yet."""

        self.end_reason = None
        """The reason why training stopped, or None if training hasn't stopped yet."""

        self.optimizer.initialize(network)

    def __evaluate_and_adjust(self, input: np.ndarray, expected_output: np.ndarray) -> list[np.ndarray]:
        # Feedforward
        h_vector_per_layer = []
        outputs_per_layer = [input]
        for i in range(self.network.layer_count):
            weights = self.network.layer_weights[i]
            activation = self.network.layer_activations[i]
            h_vector = np.matmul(outputs_per_layer[-1], weights[1:]) + weights[0]
            h_vector_per_layer.append(h_vector)
            outputs_per_layer.append(activation.primary(h_vector))

        # Backpropagation
        s_vector_per_layer = [None] * self.network.layer_count

        # For last layer
        layer_activation = self.network.layer_activations[-1]
        s_vector_per_layer[-1] = (expected_output - outputs_per_layer[-1]) * layer_activation.derivative(outputs_per_layer[-1], h_vector_per_layer[-1])

        # For inner layers
        for i in range(self.network.layer_count - 2, -1, -1):
            weights = self.network.layer_weights[i + 1]
            activation = self.network.layer_activations[i]
            s_vector_per_layer[i] = np.matmul(s_vector_per_layer[i + 1], weights[1:].T) * activation.derivative(outputs_per_layer[i + 1], h_vector_per_layer[i])

        # Calculate delta weights matrices
        dw_matrix_per_layer = [None] * self.network.layer_count
        for i in range(self.network.layer_count):
            dw_matrix_per_layer[i] = columnarize(self.optimizer.apply(i, self.learning_rate, np.concatenate((np.ones(1), outputs_per_layer[i]))[:, None] * s_vector_per_layer[i]))

        return dw_matrix_per_layer, outputs_per_layer[-1]

    def __error_for_dataset(self, dataset: list[np.ndarray], dataset_outputs: list[np.ndarray]) -> float:
        """Calculates the error for a dataset."""
        return self.error_function.error_for_dataset(dataset_outputs, [self.network.evaluate(d) for d in dataset])

    def __pretrain_check(self, dataset: list[np.ndarray], dataset_outputs: list[np.ndarray], testset: (np.ndarray | None), testset_outputs: (np.ndarray | None)):
        """Performs some pre-training checks in case some dumbass ((me)) passes in invalid values."""
        if dataset is None or dataset_outputs is None:
            raise ValueError('dataset and dataset_outputs may not be None')
        if (testset is None) != (testset_outputs is None):
            raise ValueError('To train with a testset, you must specify both testset and testset_outputs')

        if len(dataset) != len(dataset_outputs):
            raise ValueError('The lengths of dataset and dataset_output arrays must match')

        if testset is not None:
            if len(testset) != len(testset_outputs):
                raise ValueError('The lengths of testset and testset_output arrays must match')

        # In case record_error_history was changed, we create and extend the error_history array up to the current epoch.
        # This is all done to ensure self.error_history[i] contains the error value for epoch number i, even if any values were modified.
        if self.record_error_history:
            # Create error_history if it doesn't exist.
            if self.error_history is None:
                self.error_history = []
            # Extend error_history if it's not as long as the amount of epochs. Epochs that weren't recorded are filled in with None.
            if len(self.error_history) < self.current_epoch:
                # This ensures error_history is filled for epochs [0, current_epoch).
                self.error_history.extend([None] * (self.current_epoch - len(self.error_history)))
            elif len(self.error_history) > self.current_epoch:
                # If error_history is longer than it should be, cut off excess elements.
                self.error_history = self.error_history[:self.current_epoch]

            if testset is not None:
                if self.testset_error_history is None:
                    self.testset_error_history = []
                if len(self.testset_error_history) < self.current_epoch:
                    self.testset_error_history.extend([None] * (self.current_epoch - len(self.testset_error_history)))
            elif len(self.testset_error_history) > self.current_epoch:
                self.testset_error_history = self.testset_error_history[:self.current_epoch]

    def __check_update_error(self, dataset_outputs: list[np.ndarray], actual_outputs: list[np.ndarray], best_weights: list[np.ndarray],
                             testset: (np.ndarray | None), testset_outputs: (np.ndarray | None)):
        self.current_error = self.error_function.error_for_dataset(dataset_outputs, actual_outputs)
        if self.record_error_history:
            self.error_history.append(self.current_error)
            if self.testset_error_history is None:
                print(f"Error at epoch {self.current_epoch - 1}: {self.current_error}")
            else:
                self.current_testset_error = self.__error_for_dataset(testset, testset_outputs)
                self.testset_error_history.append(self.current_testset_error)
                print(f"Error at epoch {self.current_epoch - 1}: {self.current_error} with testset error {self.current_testset_error}")

        # Check if the error is better than the previously known best. If so, remember those as the current best weights found.
        if self.best_error is None or self.current_error < self.best_error:
            for i in range(len(best_weights)):
                np.copyto(best_weights[i], self.network.layer_weights[i])
            self.best_error = self.current_error

    def train(self, dataset: list[np.ndarray], dataset_outputs: list[np.ndarray], testset: (np.ndarray | None)=None, testset_outputs: (np.ndarray | None)=None):
        # Calculation variables are stored in these vectors as to reuse memory. Otherwise numpy will allocate a new vector/matrix on each operation.
        self.__pretrain_check(dataset, dataset_outputs, testset, testset_outputs)

        best_weights = self.network.layer_weights
        if self.best_error is None:
            self.best_error = self.__error_for_dataset(dataset, dataset_outputs)

        if self.current_weights is None:
            self.current_weights = [np.copy(w) for w in self.network.layer_weights]
        self.network.layer_weights = self.current_weights

        weights_adjustments = [np.zeros_like(w) for w in self.network.layer_weights]
        outputs_storage = [None] * len(dataset)
        self.end_reason = None
        while (self.end_reason is None):
            # Start next epoch.
            self.current_epoch += 1
            self.optimizer.start_next_epoch(self.current_epoch)
            for w in weights_adjustments:
                w.fill(0.0)

            # Iterate through the dataset and calculate weights adjustments for each element.
            for i in range(len(dataset)):
                adj, output = self.__evaluate_and_adjust(dataset[i], dataset_outputs[i])
                add_lists(weights_adjustments, adj)
                outputs_storage[i] = output

            # Calculate the error based on the outputs calculated while finding weight adjustments for this epoch. Since weights haven't been adjusted yet,
            # this is actually the error for the previous epoch. This is faster than calculating all the outputs again after adjusting the weights.
            self.__check_update_error(dataset_outputs, outputs_storage, best_weights, testset, testset_outputs)

            self.network.adjust_weights(weights_adjustments)

            if self.current_error <= self.acceptable_error:
                self.end_reason = 'Acceptable error reached'
            elif self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.end_reason = 'Max epochs reached'

        # Since errors for each epoch are calculated at the same time as the weight updates for the next epoch, we calculate the error for
        # the final epoch outside the loop.
        self.__check_update_error(dataset_outputs, outputs_storage, best_weights, testset, testset_outputs)

        self.current_weights = self.network.layer_weights
        self.network.layer_weights = best_weights
        print(f'Training finished with reason: {self.end_reason} after {self.current_epoch} epochs with error {self.current_error}')

    def to_json(self) -> dict:
        """Serializes this NetworkTrainer to a dict."""
        result = {
            "network": self.network.to_json(),
            "learning_rate": float(self.learning_rate),
            "error_function": self.error_function.to_json(),
            "optimizer": self.optimizer.to_json(),
            "max_epochs": int(self.max_epochs),
            "acceptable_error": float(self.acceptable_error),
            "record_error_history": float(self.record_error_history),
            "current_epoch": int(self.current_epoch)
        }

        if self.current_error is not None:
            result["current_error"] = float(self.current_error)
        if self.current_testset_error is not None:
            result["current_testset_error"] = float(self.current_testset_error)
        if self.best_error is not None:
            result["best_error"] = float(self.best_error)
        if self.end_reason is not None:
            result["end_reason"] = self.end_reason
        if self.error_history is not None:
            for i in range(len(self.error_history)):
                self.error_history[i] = float(self.error_history[i])
            result["error_history"] = self.error_history
        if self.testset_error_history is not None:
            for i in range(len(self.testset_error_history)):
                self.testset_error_history[i] = float(self.testset_error_history[i])
            result["testset_error_history"] = self.testset_error_history
        if self.current_weights is not None:
            result["current_weights"] = ndarray_list_to_json(self.current_weights)

        return result

    def save_to_file(self, file: str, indent: bool=False):
        with open(file, 'w') as f:
            json.dump(self.to_json(), f, indent=(4 if indent else None))

    def from_json(d: dict):
        t = NetworkTrainer(
            network=Network.from_json(d["network"]),
            learning_rate=float(d["learning_rate"]),
            error_function=error_function_from_json(d["error_function"]),
            optimizer=optimizer_from_json(d["optimizer"]),
            max_epochs=(int(d["max_epochs"]) if "max_epochs" in d else None),
            acceptable_error=(float(d["acceptable_error"]) if "acceptable_error" in d else None),
            record_error_history=bool(d["record_error_history"])
        )

        t.current_epoch = int(d["current_epoch"]) if "current_epoch" in d else 0
        t.current_error = float(d["current_error"]) if "current_error" in d else None
        t.current_testset_error = float(d["current_testset_error"]) if "current_testset_error" in d else None
        t.best_error = float(d["best_error"]) if "best_error" in d else None
        t.end_reason = d["end_reason"] if "end_reason" in d else None
        t.error_history = d["error_history"] if "error_history" in d else None
        t.testset_error_history = d["testset_error_history"] if "testset_error_history" in d else None
        t.current_weights = ndarray_list_from_json(d["current_weights"]) if "current_weights" in d else None
        return t

    def from_file(file: str):
        with open(file, 'r') as f:
            return NetworkTrainer.from_json(json.load(f))
