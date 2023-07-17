import sys # Python please fix your import system to work across different folders
sys.path[0] = sys.path[0].rstrip("abcdefghijklmnopqrstuvwxyz").removesuffix('/').removesuffix('\\')

import numpy as np
from src.network import Network
from src.trainer import NetworkTrainer
from src.activation import *
from src.error import *
from src.weights_initializer import *
from src.optimizer import *

digits = """
.XXX.  ,  ..X..  ,  .XXX.  ,  .XXX.  ,  ...X.  ,  XXXXX  ,  ..XX.  ,  XXXXX  ,  .XXX.  ,  .XXX. ;
X...X  ,  .XX..  ,  X...X  ,  X...X  ,  ..XX.  ,  X....  ,  .X...  ,  ....X  ,  X...X  ,  X...X ;
X..XX  ,  ..X..  ,  ....X  ,  ....X  ,  .X.X.  ,  XXXX.  ,  X....  ,  ...X.  ,  X...X  ,  X...X ;
X.X.X  ,  ..X..  ,  ...X.  ,  XXXX.  ,  X..X.  ,  ....X  ,  XXXX.  ,  ..X..  ,  .XXX.  ,  .XXXX ;
XX..X  ,  ..X..  ,  ..X..  ,  ....X  ,  XXXX.  ,  ....X  ,  X...X  ,  .X...  ,  X...X  ,  ....X ;
X...X  ,  ..X..  ,  .X...  ,  X...X  ,  ...X.  ,  X...X  ,  X...X  ,  .X...  ,  X...X  ,  ...X. ;
.XXX.  ,  .XXX.  ,  XXXXX  ,  .XXX.  ,  ...X.  ,  .XXX.  ,  .XXX.  ,  .X...  ,  .XXX.  ,  .XX..
""".replace(' ', '').replace('\n', '').replace('\r', '')

dataset = [[] for _ in range(10)]
for line in digits.split(';'):
    for i, row in enumerate(line.split(',')):
        for digit in row:
            dataset[i].append(-1 if digit == '.' else 1)

for i in range(len(dataset)):
    dataset[i] = np.asfarray(dataset[i])

arch = [
    (35, LogisticActivationFunction()),
    (10, LogisticActivationFunction()),
    (2, TanhActivationFunction()),
    (10, LogisticActivationFunction()),
    (35, SimpleActivationFunction())
]

error_function = CostAverageErrorFunction()
acceptable_error = 0

weights_init = RandomWeightsInitializer()
learning_rate = 0.01

optimizer = AdamOptimizer()

trainer = NetworkTrainer(
    network=Network(len(dataset[0]), arch, weights_init),
    learning_rate=learning_rate,
    error_function=error_function,
    optimizer=optimizer,
    max_epochs=10000,
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
    print(f"Input {i}: Expected: {dataset[i]}, got: {r} {'✅' if error_function.error_for_single(dataset[i], r) <= acceptable_error else '❌'}")

trainer.save_to_file('./usecases/networks/digits_autoencoder.nwt.json')
