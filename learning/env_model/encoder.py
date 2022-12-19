from typing import Dict, Sequence
import torch
import torch.nn as nn
import numpy as np

from ..base import BaseNN, Context
from core import Batch


class VariableEncoder(BaseNN):
    def __init__(self, context: Context):
        super().__init__(context)
        
        self.sub_modules: Dict[str, nn.Module] = {}

        for var in self.env.names_input:
            d_in = self.v(var).size
            d_h = self.dims.variable_encoder_hidden
            d_out = self.dims.variable_encoding

            self.sub_modules[var] = nn.Sequential(
                nn.Linear(d_in, d_h, **self.torchargs),
                nn.PReLU(d_h, **self.torchargs),
                nn.Linear(d_h, d_h, **self.torchargs),
                nn.LeakyReLU(),
                nn.Linear(d_h, d_out, **self.torchargs))

        for key, linear in self.sub_modules.items():
            self.add_module(f"{key}_encoder", linear)

    def forward_all(self, data: Batch):
        out = Batch(data.n)
        for var in self.sub_modules.keys():
            if var in data:
                out[var] = self.forward(var, data[var])
        return out

    def forward(self, var: str, data: torch.Tensor) -> torch.Tensor:
        sub_module = self.sub_modules[var]
        return sub_module(data)


class Aggregator(BaseNN):
    def __init__(self, dim_in: int, dim_out: int, dim_h: int, context: Context):
        super().__init__(context)

        self._dim_in = dim_in
        self._dim_out = dim_out
        self._dim_h = dim_h

        self.gru = nn.GRU(dim_in, dim_h, 1, bidirectional=True,
                          **self.torchargs)
        self.linear = nn.Linear(2 * dim_h, dim_out, **self.torchargs)

    def forward(self, seq: torch.Tensor):
        # action_seq:  len * batch * dim_a
        batch_size = seq.shape[1]

        h0 = torch.zeros(2, batch_size, self._dim_h, **self.torchargs)

        if seq.shape[0] > 0:
            self.gru.flatten_parameters()
            _, hn = self.gru(seq, h0)
        else:
            hn = h0

        hn: torch.Tensor  # 2 * batch_size * dim_h
        hn = hn.permute(1, 0, 2).reshape(batch_size, 2 * self._dim_h)
        out = self.linear(hn)
        return out


class VariableConcat(BaseNN):
    def __init__(self, context: Context, var_names: Sequence[str]):
        super().__init__(context)

        self.__names = tuple(var_names)
        self.__size = sum(self.v(k).size for k in self.__names)

    @property
    def names(self):
        return self.__names

    @property
    def size(self):
        return self.__size

    def forward(self, raw: Batch):
        to_cat = [self.raw2input(name, raw[name]) for name in self.__names]
        x = self.T.safe_cat(to_cat, (raw.n, -1), 1)
        assert x.shape[1] == self.__size
        return x