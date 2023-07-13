import sys # Python please fix your import system to work across different folders
sys.path[0] = sys.path[0].rstrip("abcdefghijklmnopqrstuvwxyz").removesuffix('/').removesuffix('\\')

import numpy as np
import matplotlib.pyplot as plt
import colorsys
from src.optimizer import *

CONFIGURATIONS = {
    'gradient': { 'optimizer': GradientDescentOptimizer(), 'learning_rage': 0.9 },
    'momentum_0.8': { 'optimizer': MomentumOptimizer(alpha=0.8), 'learning_rage': 0.9 },
    'rmsproop_0.9': { 'optimizer': RMSPropOptimizer(gamma=0.9), 'learning_rage': 0.9 },
    'adam_0.9_0.999': { 'optimizer': AdamOptimizer(beta1=0.9, beta2=0.999), 'learning_rage': 0.9 },
}

# Used as dummy object to call optimizer.initialize()
class DummyNetwork:
    def __init__(self) -> None:
        self.layer_weights = [np.array([0.0])]

def func(x):
    return np.square(x) + np.multiply(10, x) + 24 # x^2 + 10x + 24

def func_deriv(x):
    return np.multiply(2, x) + 10 # 2x + 10

minimum = (-5, func(-5))

start_x = -6.73
max_epochs = 500
acceptable_error = 0.001

results = {}
for config_name, config in CONFIGURATIONS.items():
    optimizer = config['optimizer']
    optimizer.initialize(DummyNetwork())
    learning_rate = config['learning_rage']
    x = start_x
    x_history = [x]
    gradient_history = []
    deltax_history = []
    epoch = 0

    while epoch < max_epochs and np.abs(x_history[-1] - minimum[0]) > acceptable_error:
        epoch += 1
        optimizer.start_next_epoch(epoch)
        gradient = func_deriv(x)
        deltax = -optimizer.apply(0, learning_rate, np.array([gradient]))[0]
        x = x + deltax
        x_history.append(x)
        gradient_history.append(gradient)
        deltax_history.append(deltax)
    results[config_name] = (x_history, gradient_history, deltax_history, epoch)


fig, axs = plt.subplots(nrows=1, ncols=len(CONFIGURATIONS))
i = 0
for config_name, config in CONFIGURATIONS.items():
    ax = axs.flat[i]
    x = np.linspace(minimum[0] - 2, minimum[0] + 2, 500)
    ax.plot(x, func(x))
    ax.plot(minimum[0], minimum[1], 'bo')
    i += 1
    x_history, gradient_history, deltax_history, epochs = results[config_name]
    y_history = func(x_history)
    ax.set_title(f"{config_name} eta={config['learning_rage']}")
    ax.set_xlabel(f"{epochs} epochs")
    for j in range(len(x_history) - 1):
        ax.arrow(x_history[j], y_history[j], x_history[j+1] - x_history[j], y_history[j+1] - y_history[j], length_includes_head=True, head_width=0.1, color=colorsys.hsv_to_rgb(j / len(x_history), 1, 1))

plt.show()
