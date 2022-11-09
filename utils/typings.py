from typing import Dict, Any, Union, Sequence, Tuple, Set
from torch import Tensor
from numpy import ndarray

NamedTensors = Dict[str, Tensor]
NamedArrays = Dict[str, ndarray]
NamedValues = Dict[str, Any]
SortedNames = Tuple[str, ...]
ShapeLike = Union[int, Sequence[int]]
Shape = Tuple[int, ...]
ParentDict = Dict[str, Set[str]]

SortedParentDict = Dict[str, SortedNames]
Edge = Tuple[str, str]
