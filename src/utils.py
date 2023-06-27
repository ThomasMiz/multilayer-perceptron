import numpy as np

def columnarize(x: np.ndarray):
    """If x is a row vector ([a, b, c]), returns x converted into a columnar vector [[a], [b], [c]]."""
    return x if len(x.shape) != 1 else x[:, None]

def add_lists(x: list, y: list):
    """For each element in x, adds the corresponding element in y. The result is stored in-place in x."""
    for i in range(len(x)):
        x[i] += y[i]
