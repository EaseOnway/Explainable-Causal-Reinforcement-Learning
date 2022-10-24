from typing import Any, Literal, Optional

import torch
from core import TaskInfo


class NetDims:
    def __init__(self, a=8, a_emb=16, s=16, v=16, k=16,
                    h_gru=16, h_dec=32):
        self.a = a
        self.a_emb = a_emb
        self.s = s
        self.v = v
        self.k = k
        self.h_gru = h_gru
        self.h_dec = h_dec

class NetAblations:
    def __init__(self, no_attn=False, recur=False):
        self.no_attn = no_attn
        self.recur = recur

class NetConfig:
    def __init__(self, taskinfo: TaskInfo,
                 device: Any = 'cpu',
                 dims: Optional[NetDims] = None,
                 torch_dtype: Literal['f32', 'f64'] = 'f32',
                 ablations: Optional[NetAblations] = None):
        self.dims = dims or NetDims()
        self.ablations = ablations or NetAblations()
        self.task = taskinfo
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device: torch.device = device
        self.torch_dtype = {
            'f32': torch.float32,
            'f64': torch.float64,
        }[torch_dtype]
        self.torchargs = dict(dtype=self.torch_dtype, device=self.device)

        self.__inkeys = self.task.in_state_keys | self.task.action_keys
        self.__outkeys = self.task.out_state_keys | self.task.outcomes_keys

    def var(self, key: str):
        return self.task.varinfos[key]

    @property
    def inkeys(self):
        return self.__inkeys

    @property
    def inkeys_s(self):
        return self.task.in_state_keys

    @property
    def inkeys_a(self):
        return self.task.action_keys
    
    @property
    def outkeys_s(self):
        return self.task.out_state_keys
    
    @property
    def outkeys_o(self):
        return self.task.outcomes_keys

    @property
    def outkeys(self):
        return self.__outkeys
