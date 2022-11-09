from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
from torch import Tensor, BoolTensor
import torch
import torch.distributions as D
from utils import TensorOperator as T


class Batch():

    def __init__(self, n: int, data: Optional[Dict[str, Tensor]] = None):
        self.data: Dict[str, Tensor] = data if data else {}
        self.n = n
        for value in self.data.values():
            assert isinstance(value, Tensor)
            assert value.shape[0] == n

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value: Tensor):
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

    def update(self, other: Batch):
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


class Transitions(Batch):
    RUNNING = 0
    DONE = 1
    TRUNCATED = 2

    def __init__(self, data: Dict[str, Tensor], rewards: Tensor, code: Tensor):
        n = rewards.shape[0]
        if code.shape != (n,):
            raise ValueError("wrong shape")
        super().__init__(n, data)

        self.rewards = rewards
        if code.dtype != torch.int8:
            code = code.to(torch.int8)
        self.code = code
    
    @property
    def running(self):
        return self.code == Transitions.RUNNING
    
    @property
    def done(self):
        return self.code == Transitions.DONE
    
    @property
    def truncated(self):
        return self.code == Transitions.TRUNCATED
    
    @property
    def terminated(self):
        return self.code != Transitions.RUNNING

    @staticmethod
    def from_sample(data: Dict[str, Tensor], reward: float, code: int):
        data_ = {k: v.reshape(1, *v.shape) for k, v in data.items()}
        reward_ = torch.tensor([reward], dtype=torch.float)
        code_ = torch.tensor([code], dtype=torch.int8) 
        return Transitions(data_, reward_, code_)
    
    @staticmethod
    def get_code(done: bool, truncated: bool):
        if done:
            return Transitions.DONE
        elif truncated:
            return Transitions.TRUNCATED
        else:
            return Transitions.RUNNING
    
    def to(self, device: torch.device):
        return Transitions({k: v.to(device) for k, v in self.data.items()},
                           self.rewards.to(device), self.code.to(device))


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
    
    def predict(self):
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
