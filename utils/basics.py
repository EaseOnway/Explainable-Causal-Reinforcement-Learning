from typing import Any, Iterable, Tuple

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


def select(collection: Any, indices: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(collection[i] for i in indices)


def merge_dic(d1: dict, d2: dict):
    d1 = d1.copy()
    d1.update(d2)
    return d1
