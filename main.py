import numpy as np
from src.network import Network, NetworkTrainer
from src.activation import *
from src.error import *
from src.weights_initializer import *

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

# weight_init_list = [
#     np.array([[1, 2], [3, 4]]),
#     np.array([[-0.1, -0.2, -0.3], [-2, 2, -0.2]]),
#     np.array([[5], [6], [7]])
# ]
# bias_init_list = [
#     np.array([-1, -2]),
#     np.array([-3, -4, -5]),
#     np.array([0.1])
# ]
# weight_init = ValuesWeightsInitializer(weights=weight_init_list, biases=bias_init_list)

weight_init = RandomWeightsInitializer(biases_range=(-5, 5))

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

n = Network(len(dataset[0]), arch, weight_init)

result = n.evaluate(dataset[0])
print(result)

t = NetworkTrainer(n, learning_rate, error_function)
t.train(dataset, dataset_outputs, acceptable_error)
print("IT HAS HAPPENED")
