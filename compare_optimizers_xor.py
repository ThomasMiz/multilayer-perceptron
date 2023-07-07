import numpy as np
from src.network import Network
from src.trainer import NetworkTrainer
from src.activation import *
from src.error import *
from src.weights_initializer import *
from src.optimizer import *
import matplotlib.pyplot as plt

PROBLEM_NAME = 'XOR'
RUNS_PER_OPTIMIZER = 80

CONFIGURATIONS = {
    'gradient': { 'optimizer': GradientDescentOptimizer(), 'learning_rage': 0.1 },
    'momentum_0.8': { 'optimizer': MomentumOptimizer(alpha=0.8), 'learning_rage': 0.1 },
    'rmsproop_0.9': { 'optimizer': RMSPropOptimizer(gamma=0.9), 'learning_rage': 0.01 },
    'adam_0.9_0.999': { 'optimizer': AdamOptimizer(beta1=0.9, beta2=0.999), 'learning_rage': 0.1 },
}

arch = [
    (2, SimpleActivationFunction()),
    (3, SimpleActivationFunction()),
    (1, SimpleActivationFunction())
]

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

init_weights_per_run = []
init_biases_per_run = []
for _ in range(RUNS_PER_OPTIMIZER):
    weights = [np.random.uniform(-1, 1, (len(dataset[0]) if i == 0 else arch[i-1][0], arch[i][0])) for i in range(len(arch))]
    biases = [np.random.uniform(-1, 1, arch[i][0]) for i in range(len(arch))]
    init_weights_per_run.append(weights)
    init_biases_per_run.append(biases)

error_function = CountNonmatchingErrorFunction()
acceptable_error = 0
max_epochs = 3000

results = {}

for config_name, config in CONFIGURATIONS.items():
    results[config_name] = []

for config_name, config in CONFIGURATIONS.items():
    for i in range(RUNS_PER_OPTIMIZER):
        print(f"Starting run {i+1} for {config_name}")
        n = Network(len(dataset[0]), arch, ValuesWeightsInitializer(weights=init_weights_per_run[i], biases=init_biases_per_run[i]))
        t = NetworkTrainer(network=n, learning_rate=config['learning_rage'], error_function=error_function, optimizer=config['optimizer'])
        epoch, finish_reason, error_history = t.train_analyze(dataset, dataset_outputs, acceptable_error, max_epochs)
        results[config_name].append(error_history)

for config_name, r in results.items():
    max_length = np.max([len(ri) for ri in r])
    for i in range(len(r)):
        r[i] = np.concatenate((r[i], np.zeros(max_length - len(r[i]))))
    results[config_name] = np.average(np.array(r), axis=0)

for config_name, config in CONFIGURATIONS.items():
    r = results[config_name]
    plt.semilogy(np.linspace(0, len(r)-1, len(r)), r, label=f"{config_name} eta={config['learning_rage']}")
plt.legend()
plt.title(f"Average of {RUNS_PER_OPTIMIZER} runs for {PROBLEM_NAME}")
plt.xlabel('Epoch Number')
plt.ylabel('Error Average')
plt.show()
