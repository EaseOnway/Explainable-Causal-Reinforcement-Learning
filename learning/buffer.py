from typing import Any, Dict, Union, Optional

from .base import RLBase, Context

from core import Transitions
from core.vtype import DType

from utils import Shaping
import torch
import numpy as np
from torch import Tensor


class Buffer(RLBase):

    def __init__(self, context: Context, max_size=1000):
        super().__init__(context)

        self.__max_size = max_size
        self.__cache_size = 2 * max_size
        self.__beg = 0
        self.__size = 0
        self.__data: Dict[str, Tensor] = {}
        self.__rewards = torch.empty(self.__cache_size, dtype=DType.Real.torch)
        self.__tagcodes = torch.empty(self.__cache_size, dtype=torch.int8)
        self.__tensors = _TensorGetter(self)
        self.__arrays = _ArrayGetter(self)
        self.__transitions = _TransitionGetter(self)

        for name, vtype in self.env.definition.vtypes.items():
            self.__declear(name, vtype.shape, vtype.dtype.torch)

    @property
    def max_size(self):
        return self.__max_size
    
    @property
    def rewards(self):
        return self.__rewards[self.__beg: self.__beg + self.__size]
    
    @property
    def tagcodes(self):
        return self.__tagcodes[self.__beg: self.__beg + self.__size]

    def __declear(self, name: str, shape: Shaping.Shape, dtype: torch.dtype):
        if name in self.__data:
            raise ValueError(f"data of '{name}' is already decleared")
        shape = (self.__cache_size,) + shape
        self.__data[name] = torch.empty(shape, dtype=dtype)

    def __len__(self):
        return self.__size

    def __getitem__(self, key: str) -> Tensor:
        return self.__data[key][self.__beg: self.__beg + self.__size]
    
    def keys(self):
        return self.__data.keys()

    def __contains__(self, key: str):
        return key in self.__data
    
    def __setitem__(self, name: str, data: torch.Tensor):
        if len(data) != len(self):
            raise ValueError
        
        if name not in self:
            self.__declear(name, data.shape[1:], data.dtype)

        temp = self.__data[name]
        if data.device != temp.device:
            data = data.to(device=temp.device)

        temp[self.__beg: self.__beg + self.__size] = data
    
    def __delitem__(self, name: str):
        if self.env.has_name(name):
            raise ValueError(f"cannot delete enviroment varaible {name}")
        
        del self.__data[name]
    
    @property
    def arrays(self):
        return self.__arrays
    
    @property
    def transitions(self):
        return self.__transitions
    
    @property
    def tensors(self):
        return self.__tensors
    
    def sample_batch(self, size: int):
        i = torch.randint(self.__size, size=(size,), dtype=torch.long)
        return self.transitions[i]
    
    def epoch(self, batchsize: int):
        indices = torch.randperm(self.__size)
        for i in range(0, self.__size, batchsize):
            j = min(i + batchsize, self.__size)
            if i == j:
                break
            else:
                yield self.transitions[indices[i: j]]

    def clear(self):
        self.__beg = 0
        self.__size = 0
    
    def refresh(self):
        beg, size = self.__beg, self.__size
        to_refresh = tuple(self.__data.values()) + \
            (self.__rewards, self.__tagcodes)
        for array in to_refresh:
            array[0: size] = array[beg: beg + size].clone()
        self.__beg = 0

    def write(self, data: Dict[str, Any], reward: float, tagcode: int):
        if self.__beg + self.__size >= self.__cache_size:
            self.refresh()

        data_ = self.named_tensors(data, device='cpu')

        i = self.__beg + self.__size
        for key, tensor in self.__data.items():
            try:
                value = data_[key]
            except KeyError:
                raise KeyError(f"'{key}' is missing")
            tensor[i] = value
        
        self.__rewards[i] = reward
        self.__tagcodes[i] = tagcode

        if self.__size == self.__max_size:
            self.__beg += 1
        elif self.__size < self.__max_size:
            self.__size += 1
        else:
            assert False

    def append(self, transitions: Transitions):
        if self.__beg + self.__size + transitions.n >= self.__cache_size:
            self.refresh()

        if transitions.n > self.max_size:
            raise NotImplementedError

        i = self.__beg + self.__size
        j = i + transitions.n

        for key, tensor in self.__data.items():
            try:
                value = transitions[key]
            except KeyError:
                raise KeyError(f"'{key}' is missing")
            
            tensor[i: j] = value.cpu()
        
        self.__rewards[i: j] = transitions.rewards.cpu()
        self.__tagcodes[i: j] = transitions.tagcode.cpu()

        size = j - self.__beg
        if size > self.__max_size:
            self.__beg = j - self.__max_size
            self.__size = self.__max_size
        else:
            self.__size = size
    
    def state_dict(self):
        return {'data': self.tensors[:],
                'rewards':  self.rewards[:],
                'tagcodes': self.tagcodes[:]}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        transitions = Transitions(state_dict['data'], state_dict['rewards'],
                                  state_dict['tagcodes'])
        self.append(transitions)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class _TensorGetter:
    def __init__(self, __buffer: 'Buffer'):
        self.__buffer = __buffer
    
    def __getitem__(self, index) -> Dict[str, Tensor]:
        return {k: self.__buffer[k][index]
                for k in self.__buffer.keys()}


class _ArrayGetter:
    def __init__(self, __buffer: 'Buffer'):
        self.__buffer = __buffer
    
    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        return {k: self.__buffer[k][index].numpy()
                for k in self.__buffer.keys()}


class _TransitionGetter:
    def __init__(self, __buffer: 'Buffer'):
        self.__buffer = __buffer
        self.__device = self.__buffer.device
    
    def __getitem__(self, index) -> Transitions:
        rewards = self.__buffer.rewards[index].to(self.__device)
        tagcodes = self.__buffer.tagcodes[index].to(self.__device)
        data = {k: self.__buffer[k][index].to(self.__device)
                for k in self.__buffer.keys()}

        if rewards.ndim == 0:
            return Transitions.from_sample(data, float(rewards), int(tagcodes))        
        else:
            return Transitions(data, rewards, tagcodes)
