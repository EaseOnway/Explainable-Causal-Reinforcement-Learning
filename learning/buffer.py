from typing import Any, Dict, Union

from .base import Configured
from .config import Config
from .data import Transitions
from core.vtype import VType, DType

import torch
from torch import Tensor


class Buffer(Configured):

    def __init__(self, config: Config, max_size=1000):
        super().__init__(config)

        self.__max_size = max_size
        self.__cache_size = 2 * max_size
        self.__beg = 0
        self.__size = 0
        self.__data: Dict[str, Tensor] = {}
        self.__rewards = torch.empty(self.__cache_size, dtype=DType.Numeric.torch)
        self.__dones = torch.empty(self.__cache_size, dtype=DType.Bool.torch)
        for name, vtype in self.env.info.vtypes.items():
            self.__declear(name, vtype)

    @property
    def max_size(self):
        return self.__max_size

    def __declear(self, name: str, v: VType):
        if name in self.__data:
            raise ValueError(f"data of '{name}' is already decleared")
        shape = (self.__cache_size,) + v.shape
        self.__data[name] = torch.empty(shape, dtype=v.dtype.torch)

    def __imap(self, i: Union[int, Tensor]) -> Union[int, Tensor]:
        if isinstance(i, int):
            if i < 0:
                i = self.__size + i
            return self.__beg + i
        else:
            assert isinstance(i, Tensor)
            assert i.ndim == 1
            return torch.where(i < 0, self.__beg + self.__size + i, self.__beg + i)

    def __len__(self):
        return self.__size

    def read_ats(self, indices: Tensor) -> Dict[str, Tensor]:
        i = self.__imap(indices)
        return  {name: d[i] for name, d in self.__data.items()}

    def read_at(self, index: int) -> Dict[str, Tensor]:
        i = self.__imap(index)
        assert isinstance(i, int)
        return {name: d[i] for name, d in self.__data.items()}
    
    def batch_ats(self, indices: Tensor) -> Transitions:
        i = self.__imap(indices)
        return Transitions({name: d[i].to(self.device) for name, d
                            in self.__data.items()},
                           self.__rewards[i].to(self.device),
                           self.__dones[i].to(self.device))

    def read_all(self) -> Dict[str, Tensor]:
        i = self.__imap(0)
        j = self.__imap(self.__size)
        return {name: d[i:j] for name, d in self.__data.items()}

    def batch_random(self, size: int):
        i = torch.randint(self.__size, size=(size,), dtype=torch.long)
        return self.batch_ats(i)
    
    def epoch(self, batchsize: int):
        indices = torch.randperm(self.__size)
        for i in range(0, self.__size, batchsize):
            j = min(i + batchsize, self.__size)
            if i == j:
                break
            else:
                yield self.batch_ats(indices[i: j])

    def __getitem__(self, key: str) -> Tensor:
        return self.__data[key][self.__beg: self.__beg + self.__size]
    
    @property
    def rewards(self):
        return self.__rewards[self.__beg: self.__beg + self.__size]
    
    @property
    def dones(self):
        return self.__dones[self.__beg: self.__beg + self.__size]

    def clear(self):
        self.__beg = 0
        self.__size = 0
    
    def refresh(self):
        beg, size = self.__beg, self.__size
        to_refresh = tuple(self.__data.values()) + \
            (self.__rewards, self.__dones)
        for array in to_refresh:
            array[0: size] = array[beg: beg + size]
        self.__beg = 0

    def write(self, data: Dict[str, Any], reward: float, done: bool):
        if self.__beg + self.__size >= self.__cache_size:
            self.refresh()

        data_ = self.as_raws(data, device='cpu')

        i = self.__beg + self.__size
        for key, array in self.__data.items():
            try:
                value = data_[key]
            except KeyError:
                raise KeyError(f"'{key}' is missing")
            array[i] = value
        
        self.__rewards[i] = reward
        self.__dones[i] = done

        if self.__size == self.__max_size:
            self.__beg += 1
        elif self.__size < self.__max_size:
            self.__size += 1
        else:
            assert False
