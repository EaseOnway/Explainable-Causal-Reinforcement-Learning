from typing import Dict, Optional, List, Sequence
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from .basics import Shaping
from .typings import Shape


class TensorOperator:
    def __init__(self, **torchargs):
        self.torchargs = torchargs

    @staticmethod
    def t2a(tensor: Tensor, dtype: Optional[type] = None) -> np.ndarray:
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        array: np.ndarray = tensor.numpy()
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def a2t(self, array: np.ndarray):
        x = torch.from_numpy(array).to(**self.torchargs)
        return x

    def safe_stack(self, tensors: List[Tensor], datashape: Shape):
        if len(tensors) > 0:
            return torch.stack(tensors)
        else:
            return torch.zeros((0, *datashape), **self.torchargs)

    def safe_cat(self, tensors: List[Tensor], datashape: Shape, dim: int):
        if len(tensors) > 0:
            return torch.cat(tensors, dim=dim)
        else:
            datashape = tuple((0 if d == dim else s)
                                for d, s in enumerate(datashape))
            return torch.zeros(datashape, **self.torchargs)

    @staticmethod
    def bflat(x: Tensor):
        return x.view(x.shape[0], -1)

    @staticmethod
    def bmean(x: Tensor):
        return torch.mean(x.view(x.shape[0], -1), dim=1)
    
    @staticmethod
    def bsum(x: Tensor):
        return torch.sum(x.view(x.shape[0], -1), dim=1)

    def bflatcat(self, tensors: List[Tensor], batchsize):
        return self.safe_cat([self.bflat(t) for t in tensors],
                            (batchsize, -1), 1)

    @staticmethod
    def bargmax(x: torch.Tensor):
        assert x.ndim >= 2
        batchsize = x.shape[0]
        shape = x.shape[1:]

        arg = torch.argmax(x.view(batchsize, -1), dim=1)
        out = torch.zeros(x.shape[0], len(shape), dtype=torch.long,
                        device=x.device)
        for i in range(len(shape)-1, -1, -1):
            out[:, i] = arg % shape[i]
            arg = (arg - out[:, i]) / shape[i]
        return out

    @staticmethod
    def bchoice(x: torch.Tensor):
        assert x.ndim >= 2
        batchsize = x.shape[0]
        shape = x.shape[1:]
        arg = torch.multinomial(x.view(batchsize, -1), 1).squeeze(1)
        out = torch.zeros(x.shape[0], len(shape), dtype=torch.long,
                        device=x.device)
        for i in range(len(shape)-1, -1, -1):
            out[:, i] = arg % shape[i]
            arg = (arg - out[:, i]) / shape[i]
        return out

    @staticmethod
    def valid(x: torch.Tensor):
        return not bool(torch.any(torch.isinf(x)) or torch.any(torch.isnan(x)))


class RunningStatistics:
    def __init__(self):
        self.n = 0
        self.mean: torch.Tensor
        self.__s: torch.Tensor
        self.std: torch.Tensor

    def add(self, value: torch.Tensor):
        if value.device.type != 'cpu':
            value = value.to(device = 'cpu')
        else:
            value = value.detach()

        self.n += 1
        if self.n == 1:
            self.mean = value
            self.__s = torch.zeros_like(value, device='cpu')
        else:
            old_mean = self.mean
            self.mean += (value - old_mean) / self.n
            self.__s += (value - old_mean) * (value - self.mean)
        
        self.std = torch.sqrt(self.__s / self.n)
    
    def decentralize(self, data: torch.Tensor):
        if self.n <= 1:
            return data
        else:
            return data - self.mean.to(device = data.device)
    
    def normalize(self, data: torch.Tensor):
        if self.n <= 1:
            return data
        else:
            std = torch.where(self.std == 0., 1., self.std)
            return data / std.to(device = data.device)
    
    def standardize(self, data: torch.Tensor):
        return self.normalize(self.decentralize(data))


class MultiLinear(nn.Module):
    def __init__(self, size: Sequence[int], dim_in: int, dim_out: int,
                 dtype: torch.dtype, device: torch.device, bias=True):
        '''
        size: a sequence of integars indicate the size of linear transforms.
            1 for shared transform.
        '''

        super().__init__()
        
        self.size = tuple(size)
        self.weight = nn.parameter.Parameter(torch.empty(
            *size, dim_in, dim_out,
            device=device, dtype=dtype))
        if bias:
            self.bias = nn.parameter.Parameter(torch.empty(
                *size, dim_out,
                device=device, dtype=dtype))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor):
        '''
        input: [..., *size, dim_in]
        output: [..., *size, dim_out]
        '''
        
        # x: ..., *size, dim_in
        x = x.unsqueeze(-2)  # ..., *size, 1, dim_in
        w = self.weight  # *size, dim_in, dim_out
        b = self.bias  # *size, dim_out

        y = torch.matmul(x, w)  # ..., *size, 1, dim_out
        y = y.squeeze(-2)  # ..., *size, dim_out
        
        if b is not None:
            y = y + b
        return y