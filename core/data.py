from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Union

import numpy as np
from torch import Tensor, BoolTensor
import torch
import torch.distributions as D
import enum
from utils import TensorOperator as T
from utils.typings import NamedTensors
from .env import Env


class Batch():

    def __init__(self, n: int, data: Optional[NamedTensors] = None):
        self.n = n
        self.data: Dict[str, Tensor] = {}

        if data is not None:
            for name, tensor in data.items():
                self[name] = tensor

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value: Tensor):
        if value.shape[0] != self.n:
            raise ValueError(f"the size of dim 0 should be {self.n}")
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

    def update(self, other: Union[Batch, NamedTensors]):
        for k, v in other.items():
            self[k] = v
    
    def select(self, keys: Iterable[str]):
        return Batch(self.n, {k: self.data[k] for k in keys})

    @staticmethod
    def from_sample(data: Dict[str, Tensor]):
        data_ = {k: v.reshape(1, *v.shape) for k, v in data.items()}
        return Batch(1, data_)

    def apply(self, func: Callable[[Tensor], Tensor]):
        return Batch(self.n, {k: func(v) for k, v in self.data.items()})
    
    def kapply(self, func: Callable[[str, Tensor], Tensor]):
        return Batch(self.n, {k: func(k, v) for k, v in self.data.items()})

    def to(self, device: torch.device):
        return Batch(self.n, {k: v.to(device) for k, v
                              in self.data.items()})


class Tag(enum.Enum):
    TERMINATED = 1
    TRUNCATED = 2
    INITIATED = 3

    @property
    def mask(self) -> int:
        return 1 << self.value
    
    @staticmethod
    def encode(terminated: bool, truncated: bool, initiated: bool) -> int:
        code = 0
        if terminated:
            code |= Tag.TERMINATED.mask
        if truncated:
            code |= Tag.TRUNCATED.mask 
        if initiated:
            code |= Tag.INITIATED.mask
        return code


class Transitions(Batch):
    '''Transition Batch'''

    def __init__(self, data: Dict[str, Tensor], rewards: Tensor,
                 tagcode: Tensor):
        n = rewards.shape[0]
        if tagcode.shape != (n,):
            raise ValueError("wrong shape")
        super().__init__(n, data)
        self.rewards = rewards
        if tagcode.dtype != torch.int32:
            tagcode = tagcode.to(torch.int32)
        self.tagcode = tagcode

    @property
    def truncated(self):
        return (self.tagcode & Tag.TRUNCATED.mask) != 0

    @property
    def terminated(self):
        return (self.tagcode & Tag.TERMINATED.mask) != 0
    
    @property
    def done(self):
        '''truncated or terminated'''
        return self.tagcode & (Tag.TERMINATED.mask | Tag.TRUNCATED.mask) != 0
    
    @property
    def initiated(self):
        return (self.tagcode & Tag.INITIATED.mask) != 0

    @staticmethod
    def from_sample(data: Dict[str, Tensor], reward: float, tagcode: int):
        data_ = {k: v.reshape(1, *v.shape) for k, v in data.items()}
        reward_ = torch.tensor([reward], dtype=torch.float)
        tagcode_ = torch.tensor([tagcode], dtype=torch.int32) 
        return Transitions(data_, reward_, tagcode_)
    
    def to(self, device: torch.device):
        return Transitions({k: v.to(device) for k, v in self.data.items()},
                           self.rewards.to(device), self.tagcode.to(device))
    
    def at(self, i: int):
        return Env.Transition(
           {k: v[i].cpu().numpy() for k, v in self.items()},
           float(self.rewards[i]),
           bool(self.tagcode[i] & Tag.TERMINATED.mask),
        )
    
    def iter_by_step(self):
        return iter(self.at(i) for i in range(self.n))


class Distributions:
    '''batched distribution'''

    def __init__(self, n: int,
                 distributions: Optional[Dict[str, D.Distribution]] = None):
        self.distributions = distributions or {}
        self.n = n
        for value in self.distributions.values():
            assert value.batch_shape == (n,)

    def __getitem__(self, key: str):
        return self.distributions[key]

    def __setitem__(self, key: str, value: D.Distribution):
        assert value.batch_shape[0] == self.n
        self.distributions[key] = value
    
    def __contains__(self, key: str):
        return key in self.distributions

    def keys(self):
        return self.distributions.keys()

    def items(self):
        return self.distributions.items()

    def values(self):
        return self.distributions.values()

    def iter(self):
        return iter(self.distributions)

    def update(self, other: Distributions):
        for k, v in other.items():
            self[k] = v

    def sample(self):
        return Batch(self.n, {k: d.sample() for k, d in self.items()})
    
    def mode(self):
        return Batch(self.n, {k: d.mode for k, d in self.items()})

    def logprob(self, label: Batch):
        lis = list(self.logprobs(label).values())
        return torch.sum(torch.stack(lis, dim=1), dim=1)

    def logprobs(self, label: Batch):
        return Batch(label.n, {k: T.bsum(self[k].log_prob(v))
                               for k, v in label.items()})

    def kls(self, q: Distributions):
        return Batch(q.n, {k: T.bsum(D.kl_divergence(self[k], q[k]))
                            for k in self.keys()})

    def kl(self, q: Distributions):
        kls = list(self.kls(q).values())
        return torch.sum(torch.stack(kls, dim=1), dim=1)

    def entropies(self):
        return Batch(self.n, {k: T.bsum(v.entropy())
                              for k, v in self.items()})

    def entropy(self):
        kls = list(self.entropies().values())
        return torch.sum(torch.stack(kls, dim=1), dim=1)
