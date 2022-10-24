from typing import Any, Dict, Iterable, List, Tuple
import torch


def safe_stack(tensors: List[torch.Tensor], datashape: Tuple[int, ...],
               **torchargs: Any):
    if len(tensors) > 0:
        return torch.stack(tensors)
    else:
        return torch.zeros((0, *datashape), **torchargs)

def safe_concat(tensors: List[torch.Tensor], datashape: Tuple[int, ...],
                dim: int, **torchargs: Any):
    if len(tensors) > 0:
        return torch.stack(tensors, dim=dim)
    else:
        datashape = tuple((0 if d==dim else s) for d, s in enumerate(datashape))
        return torch.zeros(datashape, **torchargs)

