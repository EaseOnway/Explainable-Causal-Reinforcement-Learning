from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Any

import torch
import torch.nn as nn

from learning.env_model.inferrer import DistributionDecoder
from learning.env_model.encoder import VariableConcat
from core.data import Batch
from utils.typings import NamedValues
import utils
from learning.config import Config
from learning.base import BaseNN


class QNet(BaseNN):
    def __init__(self, config: Config):
        super().__init__(config)

        self.varcat = VariableConcat(config, config.env.names_inputs)

        dim = config.baseline.dim_q_hidden
        self.mlp = nn.Sequential(
            nn.Linear(self.varcat.size, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, 1, **self.torchargs),
        )

    def forward(self, raw: Batch):
        x = self.varcat.forward(raw)
        q: torch.Tensor = self.mlp(x)
        q = q.squeeze(dim=0)
        return q

    def q(self, state: NamedValues, action: NamedValues):
        sa = utils.Collections.merge_dic(state, action)
        raw = Batch.from_sample(self.named_tensors(sa))
        return self.forward(raw)
