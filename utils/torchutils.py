from typing import Dict, Optional, List
import numpy as np
import torch
from torch import Tensor
from .basics import Shaping


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

    def safe_stack(self, tensors: List[Tensor], datashape: Shaping.Shape):
        if len(tensors) > 0:
            return torch.stack(tensors)
        else:
            return torch.zeros((0, *datashape), **self.torchargs)

    def safe_cat(self, tensors: List[Tensor], datashape: Shaping.Shape, dim: int):
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
