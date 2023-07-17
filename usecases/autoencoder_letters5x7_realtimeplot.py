import sys # Python please fix your import system to work across different folders
sys.path[0] = sys.path[0].rstrip("abcdefghijklmnopqrstuvwxyz").removesuffix('/').removesuffix('\\')

import numpy as np
import matplotlib.pyplot as plt
from src.network import Network
from src.trainer import NetworkTrainer
from src.weights_initializer import ValuesWeightsInitializer


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

trainer = NetworkTrainer.from_file('./usecases/networks/letters_autoencoder-35-16-2.nwt.json')
network = trainer.network

print(f"Results have an error of {trainer.error_function.error_for_dataset(dataset, [network.evaluate(d) for d in dataset])}:")
for i in range(len(dataset)):
    r = trainer.network.evaluate(dataset[i])
    print(f"Input {i} has {np.count_nonzero(dataset[i] - r)} different pixels {'✅' if trainer.error_function.error_for_single(dataset[i], r) <= trainer.acceptable_error else '❌'}")

network_arch = [(network.layer_sizes[i], network.layer_activations[i]) for i in range(network.layer_count)]
network_weights = [x[1:] for x in network.layer_weights]
network_biases = [x[0] for x in network.layer_weights]

latent_space_layer = 2

encoder = Network(
    input_size=35,
    arch=network_arch[:latent_space_layer],
    weight_initializer=ValuesWeightsInitializer(network_weights[:latent_space_layer], network_biases[:latent_space_layer])
)

decoder = Network(
    input_size=2,
    arch=network_arch[latent_space_layer:],
    weight_initializer=ValuesWeightsInitializer(network_weights[latent_space_layer:], network_biases[latent_space_layer:])
)

def plot_latents(latents, labels, a):
    a.scatter(x=[l[0] for l in latents], y=[l[1] for l in latents])
    [a.text(latents[i][0]+0.025, latents[i][1], labels[i]) for i in range(len(latents))]
    return a

def plot_image(s: np.ndarray, size: tuple[int, int], a):
    img = [[] for _ in range(size[1])]
    for y in range(size[1]):
        for x in range(size[0]):
            img[y].append(-s[x + y*size[0]])
    a.imshow(img, cmap='gray')
    return a

def interactive_latent_plot(encoder, decoder, dataset, labels):
    figs, axs = plt.subplots(nrows=1, ncols=2)
    latents = [encoder.evaluate(d) for d in dataset]
    plot_latents(latents, labels, axs[0])
    plot_image(decoder.evaluate(latents[0]), (5, 7), axs[1])
    def onclick(event):
        axs[1].set_title(f'{event.xdata}, {event.ydata}')
        if event.xdata is not None and event.ydata is not None:
            plot_image(decoder.evaluate(np.asfarray([event.xdata, event.ydata])), (5, 7), axs[1])
            plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.title("Click on the latent graph to decode that point and replot the image")
    plt.show()

interactive_latent_plot(encoder, decoder, dataset, "abcdefghijklmnopqrstuvwxyz")
