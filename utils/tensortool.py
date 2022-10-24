from typing import Union, Sequence, List, Any, Tuple
from .shaping import Shape
import numpy as np
import torch


class safe:

    @staticmethod
    def stack(tensors: List[torch.Tensor], datashape: Shape,
                   **torchargs: Any):
        if len(tensors) > 0:
            return torch.stack(tensors)
        else:
            return torch.zeros((0, *datashape), **torchargs)

    @staticmethod
    def concat(tensors: List[torch.Tensor], datashape: Shape,
                    dim: int, **torchargs: Any):
        if len(tensors) > 0:
            return torch.stack(tensors, dim=dim)
        else:
            datashape = tuple((0 if d == dim else s)
                              for d, s in enumerate(datashape))
            return torch.zeros(datashape, **torchargs)


class transform:

    @staticmethod
    def onehot_indices(array: np.ndarray) -> Tuple:
        assert array.ndim == 2
        batchsize = array.shape[0]

        temp = torch.tensor(array, dtype=torch.long)
        return (range(batchsize),) + tuple(temp[:, i] for i in range(temp.shape[1]))

    @staticmethod
    def onehot(array: np.ndarray, shape: Shape, **torchargs: Any):
        assert len(shape) == array.shape[1]
        batchsize = array.shape[0]
        index = transform.onehot_indices(array)
        x = torch.zeros(batchsize, *shape, **torchargs)
        x[index] = 1
        return x


class reduction:
    
    @staticmethod
    def batch_argmax(x: torch.Tensor):
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
