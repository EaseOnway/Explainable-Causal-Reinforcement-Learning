from typing import Any, Dict, List, Optional, Set, Tuple, final
import numpy as np
from utils import Shaping
import abc
import torch.distributions as D
import torch
from torch import Tensor


EPSILON = 1e-6


class PType(abc.ABC):
    '''Parameterized probability model
    '''

    def __init__(self, **param_dims: int):
        self.__param_dims = param_dims
    
    @property
    @final
    def param_dims(self):
        return self.__param_dims

    @abc.abstractmethod
    def __call__(self, **params: Tensor) -> D.Distribution:
        raise NotImplementedError


class Normal(PType):
    def __init__(self, dim: int, scale: Optional[float] = 1.0):
        if scale is None:
            super().__init__(mean=dim, scale=dim)
        else:
            super().__init__(mean=dim)
        
        self.scale = scale
        
    def __call__(self, mean: Tensor, scale: Optional[Tensor] = None):
        if self.scale is not None:
            _scale = self.scale
        else:
            assert scale is not None
            _scale = torch.nn.functional.softplus(scale) + EPSILON

        return D.Normal(mean, _scale)


class Categorical(PType):
    def __init__(self, k: int):
        super().__init__(weights=k)
        self.__k = k
        
    def __call__(self, weights: Tensor):
        weights = torch.softmax(weights, dim=1) + EPSILON / self.__k
        return D.Categorical(weights)
    
class Beta(PType):
    def __init__(self, dim: int):
        super().__init__(alpha=dim, beta=dim)
        
    def __call__(self, alpha: Tensor, beta: Tensor):
        a = torch.nn.functional.softplus(alpha) + EPSILON
        b = torch.nn.functional.softplus(beta) + EPSILON
        return D.Beta(a, b)


class Bernoulli(PType):
    def __init__(self):
        super().__init__(logit=1)
        
    def __call__(self, logit: Tensor):
        p = torch.sigmoid(logit)
        return D.Bernoulli(p)
