import numpy as np
from src.network import Network
from src.trainer import NetworkTrainer
from src.activation import *
from src.error import *
from src.weights_initializer import *
from src.optimizer import *

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

"""
# Initialize weights and biases with predefined values (these ones work even without an optimizer).
weight_init_list = [
    np.array([[1, 2], [3, 4]]),
    np.array([[-0.1, -0.2, -0.3], [-2, 2, -0.2]]),
    np.array([[5], [6], [7]])
]
bias_init_list = [
    np.array([-1, -2]),
    np.array([-3, -4, -5]),
    np.array([0.1])
]
weight_init = ValuesWeightsInitializer(weights=weight_init_list, biases=bias_init_list)

optimizer = GradientDescentOptimizer()

"""

"""
# Initialize weights and biases with prefedined values that don't work (at least not without an optimizer).
weight_init_list = [
    np.array([
        [-0.02912798, 0.14168438],
        [-0.00282928, -0.00574109]
    ]),
    np.array([
        [-0.02389368, 0.12952779, -0.0980236],
        [-0.04719204, 0.09316212, 0.05974225]
    ]),
    np.array([
        [-0.3476737],
        [-0.25117512],
        [-0.10442769]
    ])
]

bias_init_list = [
    np.array([0.00319152, 0.12801749]),
    np.array([0.13535344, 0.04262677, 0.10478634]),
    np.array([0.02001691])
]

optimizer = GradientDescentOptimizer()

weight_init = ValuesWeightsInitializer(weights=weight_init_list, biases=bias_init_list)
"""

# Initialize weights and biases with random values in a range.
weight_init = RandomWeightsInitializer(biases_range=(-5, 5))
# optimizer = GradientDescentOptimizer() # Random weights don't always work with no optimizer (that is, with gradient descent)
# optimizer = MomentumOptimizer(alpha=0.8) # Momentum fixes this issue
optimizer = RMSPropOptimizer(gamma=0.9)

error_function = CountNonmatchingErrorFunction()
acceptable_error = 0

learning_rate = 0.1


n = Network(len(dataset[0]), arch, weight_init)

print("Initial results:")
for i in range(len(dataset)):
    r = n.evaluate(dataset[i])
    print(f"Input {i}: Expected: {dataset_outputs[i]}, got: {r} {'✅' if np.array_equal(r, dataset_outputs[i]) else '❌'}")

print("\nTraining...")
t = NetworkTrainer(n, learning_rate, error_function, optimizer)
t.train(dataset, dataset_outputs, acceptable_error)

print("\nFinal results:")
for i in range(len(dataset)):
    r = n.evaluate(dataset[i])
    print(f"Input {i}: Expected: {dataset_outputs[i]}, got: {r} {'✅' if np.array_equal(r, dataset_outputs[i]) else '❌'}")
