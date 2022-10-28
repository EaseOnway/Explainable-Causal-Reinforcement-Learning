from __future__ import annotations
from typing import Any, Callable, Dict, Generic, Iterable, Optional, TypeVar, Union

from .scm import StructrualCausalModel
from .env import VarInfo
import utils.tensorfuncs as T

import numpy as np
import torch


_tdata = TypeVar('_tdata', torch.Tensor, np.ndarray)


class Batch(Generic[_tdata]):

    def __init__(self, n: int, data: Optional[Dict[str, _tdata]] = None):
        self.data: Dict[str, _tdata] = data if data else {}
        self.n = n
        for value in self.data.values():
            assert value.shape[0] == n

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value: _tdata):
        assert value.shape[0] == self.n
        self.data[key] = value
    
    def __contains__(self, key: str):
        return key in self.data

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def iter(self):
        return iter(self.data)

    def update(self, other: Batch[_tdata]):
        for k, v in other.items():
            self[k] = v
    
    def select(self, keys: Iterable[str]):
        return Batch(self.n, {k: self.data[k] for k in keys})

    @staticmethod
    def from_sample(datadic: Dict[str, _tdata]):
        d = {k: v.reshape(1, *v.shape) for k, v in datadic.items()}
        return Batch(1, d)

    def apply(self, func: Callable[[str, _tdata,], _tdata]):
        return Batch(self.n, {k: func(k, v) for k, v
                              in self.data.items()})
    
    @staticmethod
    def torch(n: int, data: Optional[Dict[str, torch.Tensor]] = None):
        return Batch(n, data)
    
    @staticmethod
    def numpy(n: int, data: Optional[Dict[str, np.ndarray]] = None):
        return Batch(n, data)


class Buffer():
    def __init__(self, varinfos: Dict[str, VarInfo], max_size=1000):
        self.__max_size = max_size
        self.__beg = 0
        self.__size = 0
        self.__data: Dict[str, np.ndarray] = {}
        self.__varinfos = varinfos

        for name, varinfo in varinfos.items():
            self.__declear(name, varinfo)

    @property
    def max_size(self):
        return self.__max_size

    def shape_of(self, name: str):
        return self.__data[name].shape[1:]

    def type_of(self, name: str):
        return self.__data[name].dtype

    def __declear(self, name: str, varinfo: VarInfo):
        if name in self.__data:
            raise ValueError(f"data of '{name}' is already decleared")
        shape = (2*self.__max_size,) + varinfo.shape
        self.__data[name] = np.empty(shape, varinfo.dtype)

    def __imap(self, i: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        if isinstance(i, int):
            if i < 0:
                i = self.__size + i
            return self.__beg + i
        else:
            assert isinstance(i, np.ndarray)
            assert i.ndim == 1
            return np.where(i < 0, self.__beg + self.__size + i, self.__beg + i)

    def __len__(self):
        return self.__size

    def read_ats(self, indices: np.ndarray) -> Batch[np.ndarray]:
        i = self.__imap(indices)
        assert isinstance(i, np.ndarray)
        return Batch(len(i), {name: d[i] for name, d in self.__data.items()})

    def read_at(self, index: int) -> Dict[str, np.ndarray]:
        i = self.__imap(index)
        assert isinstance(i, int)
        return {name: d[i] for name, d in self.__data.items()}

    def read_all(self) -> Batch[np.ndarray]:
        i = self.__imap(0)
        j = self.__imap(self.__size)
        return Batch(self.__size, {name: d[i:j] for name, d
                                   in self.__data.items()})

    def sample_batch(self, size: int, replace=True):
        i = np.random.choice(self.__size, size, replace=replace)
        return self.read_ats(i)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.__data[key][self.__beg: self.__beg + self.__size]

    def clear(self):
        self.__beg = 0
        self.__size = 0

    def write(self, data: Dict[str, Any]):
        if self.__beg + self.__size >= self.__max_size * 2:
            for array in self.__data.values():
                array[:self.__size] = array[self.__beg:]
            self.__beg = 0

        i = self.__beg + self.__size
        for key, array in self.__data.items():
            try:
                value = data[key]
            except KeyError:
                value = self.__varinfos[key].default
                if value is None:
                    raise KeyError(
                        f"'{key}' is not given, and has no default value.")

            array[i] = value

        if self.__size == self.__max_size:
            self.__beg += 1
        elif self.__size < self.__max_size:
            self.__size += 1
        else:
            assert False
