import numpy as np
from src.network import Network, NetworkTrainer
from src.activation import *
from src.error import *
from src.weights_initializer import *

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
            dataset[i].append(0 if digit == '.' else 1)
for i in range(len(dataset)):
    dataset[i] = np.array(dataset[i])

dataset_outputs = []
for i in range(10):
    d = [-1] * 10
    d[i] = 1
    dataset_outputs.append(np.array(d))

arch = [
    (24, TanhActivationFunction({'beta': 1})),
    (20, TanhActivationFunction({'beta': 1})),
    (10, SimpleActivationFunction())
]

error_function = CountNonmatchingErrorFunction()
acceptable_error = 0

weights_init = RandomWeightsInitializer()
learning_rate = 0.01

n = Network(len(dataset[0]), arch, weights_init)

result = n.evaluate(dataset[0])
print(result)

t = NetworkTrainer(n, learning_rate, error_function)
t.train(dataset, dataset_outputs, acceptable_error)
print("IT HAS HAPPENED")

for i in range(len(dataset)):
    r = n.evaluate(dataset[i])
    print(f"Input {i}: Expected: {dataset_outputs[i]}, got: {r} {'✅' if np.array_equal(r, dataset_outputs[i]) else '❌'}")
