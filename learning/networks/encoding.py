from typing import Dict, Sequence
import torch
import torch.nn as nn
import numpy as np

from .config import NetConfig
from .base import BaseNN


class VariableEncoder(BaseNN):
    def __init__(self, dim: int, vars: Sequence[str], config: NetConfig):
        super().__init__(config)
        self.sub_modules: Dict[str, nn.Module] = {}

        for var in vars:
            size = self.config.var(var).size
            size1 = min(2 * size + len(vars), 4 * dim)
            size2 = min(2 * size1, 4 * dim)

            self.sub_modules[var] = nn.Sequential(
                nn.Linear(size, size1, **config.torchargs),
                nn.PReLU(size1, **config.torchargs),
                nn.Linear(size1, size2, **config.torchargs),
                nn.LeakyReLU(),
                nn.Linear(size2, dim, **config.torchargs),
            )

        for key, linear in self.sub_modules.items():
            self.add_module(f"{key}_encoder", linear)

    def forward(self, datadic: Dict[str, np.ndarray]):
        out: Dict[str, torch.Tensor] = {}
        for var, linear in self.sub_modules.items():
            data = datadic[var]
            x = torch.from_numpy(data.reshape(
                len(data), -1)).to(**self.torchargs)
            out[var] = linear(x)
        return out


class Aggregator(BaseNN):
    def __init__(self, dim_in: int, dim_out: int, config: NetConfig):
        super().__init__(config)

        dim_h = self.dims.h_gru
        self._dim_in = dim_in
        self._dim_out = dim_out

        self.gru = nn.GRU(dim_in, dim_h, 1, bidirectional=True,
                          **config.torchargs)
        self.linear = nn.Linear(2 * dim_h, dim_out, **config.torchargs)

    def forward(self, seq: torch.Tensor):
        # action_seq:  len * batch * dim_a
        batch_size = seq.shape[1]
        dim_h = self.dims.h_gru

        h0 = torch.zeros(2, batch_size, dim_h, **self.torchargs)

        if seq.shape[0] > 0:
            _, hn = self.gru(seq, h0)
        else:
            hn = h0

        hn: torch.Tensor  # 2 * batch_size * dim_h
        hn = hn.permute(1, 0, 2).reshape(batch_size, 2 * dim_h)
        out = self.linear(hn)
        return out