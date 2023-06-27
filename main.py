import numpy as np
from src.network import Network, NetworkTrainer
from src.activation import *
from src.error import *

dataset = [
    np.array([1, 1]),
    np.array([1, -1]),
    np.array([-1, 1]),
    np.array([-1, -1]),
]

dataset_outputs = [
    np.array([1]),
    np.array([-1]),
    np.array([-1]),
    np.array([1]),
]

arch = [
    (2, SimpleActivationFunction()),
    (3, SimpleActivationFunction()),
    (1, SimpleActivationFunction())
]

error_function = CountNonmatchingErrorFunction()
acceptable_error = 0

learning_rate = 0.1

"""

dataset = [
    np.array([0.5, 0.1, -0.2]),
]

dataset_outputs = [
    np.array([0.6])
]

arch = [
    (2, LogisticActivationFunction({'beta': 0.5})),
    (1, LogisticActivationFunction({'beta': 0.5}))
]

error_function = CostAverageErrorFunction()
acceptable_error = 1e-10

learning_rate = 0.5

"""

n = Network(len(dataset[0]), arch)
# print(n.layer_weights)
# print(n.layer_weights[0].shape)
# n.layer_weights = [np.array([[0.5, -0.6], [0.1, -0.2], [0.1, 0.7]]), np.array([[0.1], [-0.3]])]
# print(n.layer_weights)


result = n.evaluate(dataset[0])
print(result)

t = NetworkTrainer(n, learning_rate, error_function)
t.train(dataset, dataset_outputs, acceptable_error)
print("IT HAS HAPPENED")
