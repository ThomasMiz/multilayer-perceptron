import sys # Python please fix your import system to work across different folders
sys.path[0] = sys.path[0].rstrip("abcdefghijklmnopqrstuvwxyz").removesuffix('/').removesuffix('\\')

import numpy as np
from src.network import Network
from src.trainer import NetworkTrainer
from src.activation import *
from src.error import *
from src.weights_initializer import *
from src.optimizer import *


patterns = {}
with open('usecases/datasets/characters5x7.txt', 'r') as file:
    rawdata = file.read().replace('\n', '').replace('\r', '').split(':')

for entry in rawdata:
    entry = entry.strip()
    if len(entry) == 0:
        continue
    e = entry.split('=')
    name = e[0].strip()
    patterns[name] = np.asfarray([1.0 if c == 'X' else -1.0 for c in e[1]])

dataset = [patterns[name] for name in "abcdefghijklmnopqrstuvwxyz"]

arch = [
    (16, LogisticActivationFunction()),
    (2, TanhActivationFunction()),
    (16, LogisticActivationFunction()),
    (35, SimpleActivationFunction())
]

error_function = CostAverageErrorFunction()
acceptable_error = 0

weights_init = RandomWeightsInitializer()
learning_rate = 0.001

optimizer = AdamOptimizer()

trainer = NetworkTrainer(
    network=Network(len(dataset[0]), arch, weights_init),
    learning_rate=learning_rate,
    error_function=error_function,
    optimizer=optimizer,
    max_epochs=1000000,
    acceptable_error=acceptable_error
)

print(f"Initial results have an error of {error_function.error_for_dataset(dataset, [trainer.network.evaluate(d) for d in dataset])}:")
for i in range(len(dataset)):
    r = trainer.network.evaluate(dataset[i])
    print(f"Input {i}: Expected: {dataset[i]}, got: {r} {'✅' if error_function.error_for_single(dataset[i], r) <= acceptable_error else '❌'}")

print("\nTraining...")
trainer.train(dataset, dataset)

print(f"\nFinal results have an error of {trainer.best_error}:")
for i in range(len(dataset)):
    r = trainer.network.evaluate(dataset[i])
    #print(f"Input {i}: Expected: {dataset[i]}, got: {r} {'✅' if error_function.error_for_single(dataset[i], r) <= acceptable_error else '❌'}")
    print(f"Input {i} has {np.count_nonzero(dataset[i] - r)} different pixels {'✅' if error_function.error_for_single(dataset[i], r) <= acceptable_error else '❌'}")

trainer.save_to_file('./usecases/networks/letters_autoencoder-35-16-2.nwt.json')
