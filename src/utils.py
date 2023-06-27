import numpy as np

def columnarize(x: np.ndarray):
    """If x is a row vector ([a, b, c]), returns x converted into a columnar vector [[a], [b], [c]]"""
    return x if len(x.shape) != 1 else x[:, None]
