from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from typing import Optional, Union, Sequence, List, Any, Tuple
from utils.shaping import Shape
import numpy as np
import torch


def t2a(tensor: torch.Tensor, dtype: Optional[type] = None):
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    array: np.ndarray = tensor.numpy()
    if dtype is not None:
        array = array.astype(dtype)
    return array

def a2t(array: np.ndarray, **torchargs):
    x = torch.from_numpy(array).to(**torchargs)
    return x

def safe_stack(tensors: List[torch.Tensor], datashape: Shape,
                **torchargs: Any):
    if len(tensors) > 0:
        return torch.stack(tensors)
    else:
        return torch.zeros((0, *datashape), **torchargs)

def safe_cat(tensors: List[torch.Tensor], datashape: Shape,
            dim: int, **torchargs: Any):
    if len(tensors) > 0:
        return torch.cat(tensors, dim=dim)
    else:
        datashape = tuple((0 if d == dim else s)
                            for d, s in enumerate(datashape))
        return torch.zeros(datashape, **torchargs)

def batch_flat(x: torch.Tensor):
    return x.view(x.shape[0], -1)

def batch_flatcat(tensors: List[torch.Tensor], batchsize,
                **torchargs: Any):
    return safe_cat([batch_flat(t) for t in tensors],
                        (batchsize, -1), 1)

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

def batch_choice(x: torch.Tensor):
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
