import numpy as np
import binascii


def columnarize(x: np.ndarray):
    """If x is a row vector ([a, b, c]), returns x converted into a columnar vector [[a], [b], [c]]."""
    return x if len(x.shape) != 1 else x[:, None]


def add_lists(x: list, y: list):
    """For each element in x, adds the corresponding element in y. The result is stored in-place in x."""
    for i in range(len(x)):
        np.add(x[i], y[i], out=x[i])


def ndarray_to_json(x: np.ndarray) -> dict:
    if x.ndim == 0 or x.ndim > 2:
        raise ValueError('Cannot convert ndarray with 0 or more than 2 dimentions to JSON')
    x = np.asfarray(x)
    result = {}
    result["rows"] = x.shape[0]
    if x.ndim > 1:
        result["columns"] = x.shape[1]
    result["rawbytes"] = str(binascii.hexlify(x.tobytes()), 'ascii')
    return result


def ndarray_from_json(d: dict) -> np.ndarray:
    shape = (int(d["rows"]), int(d["columns"])) if "columns" in d else (int(d["rows"]), )
    return np.copy(np.frombuffer(binascii.unhexlify(d["rawbytes"]), dtype=np.float64).reshape(shape))


def ndarray_list_to_json(x: list[np.ndarray]) -> list:
    return [ndarray_to_json(xi) for xi in x]


def ndarray_list_from_json(x: list) -> list[np.ndarray]:
    return [ndarray_from_json(xi) for xi in x]
