from typing import Any, Iterable

import numpy as np


def argmax(collection: Any, indices: Iterable[Any]):
    im = None
    m = None
    for i in indices:
        v = collection[i]
        if m is None or v > m:
            im, m = i, v
    return im, m


def argmin(collection: Any, indices: Iterable[Any]):
    im = None
    m = None
    for i in indices:
        v = collection[i]
        if m is None or v < m:
            im, m = i, v
    return im, m


def prob(p: float):
    return np.random.rand() <= p