from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Any

import torch
import torch.nn as nn

from learning.env_model.modules import DistributionDecoder
from learning.buffer import Buffer
from learning.base import BaseNN, Context
from learning.planning.ppo import PPO, Critic, Actor
from core import Batch, Transitions, Distributions
from utils.typings import NamedValues
from core import DType
import utils as u


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

    def forward(self, raw: Batch, mask: Optional[torch.Tensor] = None):
        if mask is None:
            to_cat = [self.raw2input(name, raw[name]) for name in self.__names]
        else:
            to_cat = [self.raw2input(name, raw[name]) * mask[:, i].unsqueeze(1)
                      for i, name in enumerate(self.__names)]

        x = self.T.safe_cat(to_cat, (raw.n, -1), 1)
        assert x.shape[1] == self.__size
        return x


class StateEncoder(VariableConcat):
    def __init__(self, context: Context,
                 restriction: Optional[Iterable[str]] = None):
        if restriction is None:
            names = context.env.names_s
        else:
            names = sorted(set(restriction))
        super().__init__(context, names)

        dim = self.dims.actor_critic_hidden
        self.mlp = nn.Sequential(
            nn.Linear(self.size, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
            nn.LeakyReLU(),
        )

    def forward(self, raw: Batch, mask: Optional[torch.Tensor] = None):
        x = super().forward(raw, mask)
        e: torch.Tensor = self.mlp(x)
        return e


class SaliencyActor(Actor):
    def __init__(self, context: Context):
        BaseNN.__init__(self, context)

        self._akeys = self.env.names_a

        self.encoder = StateEncoder(context)

        dim = self.dims.actor_critic_hidden
        self.mask_decoder = nn.Sequential(
            nn.Linear(dim, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, self.env.num_s, **self.torchargs),
            nn.Sigmoid()
        )
        self.decoders = {var: self.__make_decoder(var)
                         for var in self._akeys}

    def __make_decoder(self, var: str):
        decoder = DistributionDecoder(self.dims.actor_critic_hidden,
                                      self.v(var), self.context)
        self.add_module(f'{var} decoder', decoder)
        return decoder

    def forward(self, raw: Batch):
        e = self.encoder.forward(raw)
        mask: torch.Tensor = self.mask_decoder(e)
        e = self.encoder.forward(raw, mask)
        out = Distributions(raw.n)
        for k, d in self.decoders.items():
            da = d.forward(e)
            out[k] = da

        self.mask = mask
        return out
    
