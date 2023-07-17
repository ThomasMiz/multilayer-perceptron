import sys # Python please fix your import system to work across different folders
sys.path[0] = sys.path[0].rstrip("abcdefghijklmnopqrstuvwxyz").removesuffix('/').removesuffix('\\')

import csv
import numpy as np
from src.network import Network
from src.trainer import NetworkTrainer
from src.activation import *
from src.error import *
from src.weights_initializer import *
from src.optimizer import *

dataset = []
dataset_outputs = []

with open('usecases/datasets/heart_disease.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    i = 0
    for row in reader:
        if i == 0:
            print(f"Parsing CSV: {row}")
        else:
            dataset.append(np.asfarray(row[:-1]))
            dataset_outputs.append(np.asfarray([float(row[-1]) / 4.0 * 0.9 + 0.05])) # Convert the expected outputs from range (0, 4) to (-0.8, 0.8)
        i += 1

# Normalize the dataset so there are no large numbers by transforming everything to a range (-1, 1)
mins = np.min(dataset, axis=0)
maxs = np.max(dataset, axis=0)
m = 2.0 / (maxs - mins)
for d in dataset:
    np.subtract(d, mins, out=d)
    np.multiply(d, m, out=d)
    np.subtract(d, 1, out=d)

arch = [
    (128, LogisticActivationFunction()),
    (96, LogisticActivationFunction()),
    (64, LogisticActivationFunction()),
    (32, LogisticActivationFunction()),
    (16, LogisticActivationFunction()),
    (1, LogisticActivationFunction())
]

error_function = CostAverageErrorFunction()
acceptable_error = 0.1

weights_init = RandomWeightsInitializer()
learning_rate = 0.001

optimizer = AdamOptimizer()

network = Network(len(dataset[0]), arch, weights_init)

trainer = NetworkTrainer(
    network=network,
    learning_rate=learning_rate,
    error_function=error_function,
    optimizer=optimizer,
    max_epochs=1000,
    acceptable_error=acceptable_error,
    record_error_history=True
)

trainer.train(dataset, dataset_outputs)

trainer.save_to_file('./usecases/networks/heart_disease.nwt.json')

for i in range(len(dataset)):
    r = network.evaluate(dataset[i])
    err = error_function.error_for_single(dataset_outputs[i], r)
    print(f"Input {i}: Expected: {dataset_outputs[i]}, got: {r} {'✅' if err <= acceptable_error else '❌'}")
