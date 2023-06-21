import numpy as np
from src.network import Network, NetworkTrainer
from src.theta import ThetaFunction, SimpleThetaFunction, LinealThetaFunction, TanhThetaFunction, LogisticThetaFunction

dataset = [
    np.array([1, 1]),
    np.array([1, -1]),
    np.array([-1, 1]),
    np.array([-1, -1]),
]

dataset_outputs_xor = [
    np.array([1]),
    np.array([-1]),
    np.array([-1]),
    np.array([1]),
]

arch = [
    (2, SimpleThetaFunction()),
    (3, SimpleThetaFunction()),
    (1, SimpleThetaFunction())
]
eta = 0.1

"""

dataset = [
    np.array([0.5, 0.1, -0.2]),
]

dataset_outputs_xor = [
    np.array([0.6])
]

arch = [
    (2, LogisticThetaFunction({'beta': 0.5})),
    (1, LogisticThetaFunction({'beta': 0.5}))
]

eta = 0.5

"""

n = Network(len(dataset[0]), arch)
print(n.weights_per_layer)
print(n.weights_per_layer[0].shape)
#n.weights_per_layer = [np.array([[0.5, -0.6], [0.1, -0.2], [0.1, 0.7]]), np.array([[0.1], [-0.3]])]
print(n.weights_per_layer)


result = n.evaluate(dataset[0])
print(result)

t = NetworkTrainer(n, eta)
t.train(dataset, dataset_outputs_xor)
print("IT HAS HAPPENED")
