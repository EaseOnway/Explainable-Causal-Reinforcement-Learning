from typing import Sequence, Tuple, Optional
import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from utils import MultiLinear
from utils.typings import SortedNames
from core import Batch
from learning.config import Config
from ..base import BaseNN, Context
from core.vtype import VType


class DistributionDecoder(BaseNN):
    def __init__(self, dim_in: int, vtype: VType, context: Context):
        super().__init__(context)
        self._vtype = vtype
        self._ptype = vtype.ptype

        dh_dec = self.dims.decoder_hidden

        self.sub_decoders = {key: nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim_in, dh_dec, **self.torchargs),
            nn.PReLU(dh_dec, **self.torchargs),
            nn.Linear(dh_dec, dim, **self.torchargs),
        ) for key, dim in self._ptype.param_dims.items()}
        for param, decoder in self.sub_decoders.items():
            self.add_module(f"{param} decoder", decoder)

    def forward(self, x):
        params = {k: decoder(x) for k, decoder in self.sub_decoders.items()}
        out = self._ptype(**params)
        return out


class VariableEncoder(BaseNN):
    def __init__(self, context, names: SortedNames):
        super().__init__(context)

        self.names = names
        self.dim_in = dim_in = sum(self.v(name).size for name in names)
        self.dim_h = dim_h = self.dims.variable_encoder_hidden
        self.dim_out = dim_out = self.dims.variable_encoding

        self.linear1 = nn.Linear(dim_in, dim_h, **self.torchargs)
        self.readout = nn.Sequential(
            nn.LeakyReLU(),
            MultiLinear([len(names)], dim_h, dim_h, **self.torchargs),
            nn.LeakyReLU(),
            MultiLinear([len(names)], dim_h, dim_out, **self.torchargs),
            nn.Tanh()
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        '''
        input: data of named variables.
        output: tensor(batch, n_variable, dim_encoding).
        '''
        x0: torch.Tensor = torch.block_diag(
            *(batch[name] for name in self.names))
        x1: torch.Tensor = self.linear1(x0)
        x2 = x1.reshape(len(self.names), batch.n, self.dim_h).transpose(0, 1)
        x3: torch.Tensor = self.readout(x2)
        return x3


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
