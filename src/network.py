import numpy as np
from src.theta import ThetaFunction

class Network:
    def __init__(self, input_size: int, arch: list[tuple[int, ThetaFunction]]) -> None:
        if len(arch) < 2:
            raise ValueError("A network must have at least two layers")

        self.input_size = input_size
        self.arch = arch

        # Stores the weights for all layers. Each layer's weights are represented by a matrix
        # Each matrix's size is decided by sizes of it's respective layer and it's previous layer (or input, for the first layer)
        self.weights_per_layer = []
        for i in range(len(arch)):
            # Initialize all weights as random values between -1 and 1
            prev_layer_size = input_size if i == 0 else arch[i - 1][0]
            self.weights_per_layer.append(np.random.rand(prev_layer_size, arch[i][0]) * 2 - 1)

    def evaluate(self, input):
        input = np.array(input)
        if input.ndim != 1:
            raise ValueError("The input must have only 1 dimention")
        if len(input) != self.input_size:
            raise ValueError("The input size must match the network's input size")

        states_per_layer = [input]
        for layer_weights in self.weights_per_layer:
            states_per_layer.append(np.matmul(states_per_layer[-1], layer_weights))
        return states_per_layer[-1]
