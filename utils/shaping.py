from typing import Tuple, Union, Sequence


ShapeLike = Union[int, Sequence[int]]
Shape = Tuple[int, ...]


def as_shape(shape: ShapeLike) -> Shape:
    if isinstance(shape, int):
        return (shape,)
    else:
        return tuple(shape)

def get_size(shape: ShapeLike):
    if isinstance(shape, int):
        return shape
    else:
        s = 1
        for d in shape:
            s *= d
        return s
