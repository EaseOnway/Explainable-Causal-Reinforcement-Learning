from typing import Any, Iterable, Tuple, Union, Sequence
from .typings import ShapeLike, Shape

import numpy as np


class Collections:
    @staticmethod
    def argmax(collection: Any, indices: Iterable[Any]):
        im = None
        m = None
        for i in indices:
            v = collection[i]
            if m is None or v > m:
                im, m = i, v
        return im, m

    @staticmethod
    def argmin(collection: Any, indices: Iterable[Any]):
        im = None
        m = None
        for i in indices:
            v = collection[i]
            if m is None or v < m:
                im, m = i, v
        return im, m

    @staticmethod
    def select(collection: Any, indices: Iterable[Any]) -> Tuple[Any, ...]:
        return tuple(collection[i] for i in indices)

    @staticmethod
    def merge_dic(d1: dict, d2: dict):
        d1 = d1.copy()
        d1.update(d2)
        return d1


class Random:

    @staticmethod
    def event(p: float):
        return np.random.rand() <= p


class Shaping:

    Shape = Shape
    ShapeLike = ShapeLike

    @staticmethod
    def as_shape(shape: ShapeLike) -> Shape:
        if isinstance(shape, int):
            return (shape,)
        else:
            return tuple(shape)

    @staticmethod
    def get_size(shape: ShapeLike) -> int:
        if isinstance(shape, int):
            return shape
        else:
            return np.prod(shape, dtype=int)
