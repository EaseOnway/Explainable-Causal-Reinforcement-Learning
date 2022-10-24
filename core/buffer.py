from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


from .scm import StructrualCausalModel
from .taskinfo import TaskInfo, VarInfo

import numpy as np


class Buffer():
    def __init__(self, varinfos: Dict[str, VarInfo], max_size = 1000):
        self.__max_size = max_size
        self.__beg = 0
        self.__size = 0
        self.__data: Dict[str, np.ndarray] = {}
        self.__defaults: Dict[str, Any] = {}

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
        shape, dtype, default = varinfo.shape, varinfo.dtype, varinfo.default
        if name in self.__data:
            raise ValueError(f"data of '{name}' is already decleared")
        if isinstance(shape, int):
            shape = (shape,)
        if default is not None:
            self.__defaults[name] = default
        data_shape = (2*self.__max_size,) + shape
        self.__data[name] = np.empty(shape=data_shape, dtype=dtype)

    def __imap(self, i: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        if isinstance(i, int):
            if i < 0:
                i = self.__size + i
            return self.__beg + i
        else:
            assert isinstance(i, np.ndarray)
            return np.where(i < 0, self.__beg + self.__size + i, self.__beg + i)
    
    def __slicemap(self, s: slice):
        start = s.start or 0
        stop = s.stop or self.__size
        step: Optional[int] = s.step
        return slice(self.__imap(start), self.__imap(stop), step)
    
    def __len__(self):
        return self.__size
    
    def __getitem__(self, index) -> Dict[str, np.ndarray]:  # type: ignore
        if isinstance(index, int) or isinstance(index, np.ndarray):
            i = self.__imap(index)
            return {name: d[i] for name, d in self.__data.items()}
        elif isinstance(index, slice):
            i = self.__slicemap(index)
            return {name: d[i] for name, d in self.__data.items()}
        else:
            TypeError(f'index type {type(index)} not supported')
    
    def read(self):
        return self[:]
    
    def sample_batch(self, size: int, replace=True):
        i = np.random.choice(self.__size, size, replace=replace)
        return self[i]

    def get(self, key: str) -> np.ndarray:
        return self.__data[key][self.__beg: self.__beg + self.__size]

    def clear(self):
        self.__beg = 0
        self.__size = 0
    
    def write(self, **kargs):
        if self.__beg + self.__size >= self.__max_size * 2:
            for array in self.__data.values():
                array[:self.__size] = array[self.__beg:]
            self.__beg = 0
        
        i = self.__beg + self.__size
        for key, array in self.__data.items():
            try:
                value = kargs[key]
            except KeyError:
                try:
                    value = self.__defaults[key]
                except KeyError:
                    raise KeyError(f"'{key}' is not given, and has no default value.")

            array[i] = value

        if self.__size == self.__max_size:
            self.__beg += 1
        elif self.__size < self.__max_size:
            self.__size += 1
        else:
            assert False
