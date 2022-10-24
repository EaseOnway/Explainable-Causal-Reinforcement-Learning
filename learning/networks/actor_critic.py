from typing import Dict
import torch
import torch.nn as nn
import numpy as np
from .inferrer import Inferrer
from .config import *
import utils as u
from .base import BaseNN


class Critic(BaseNN):
    def __init__(self, config: NetConfig):
        super().__init__(config)

        self._outcome_keys = tuple(config.task.outcomes_keys)
        self._outcome_weights = torch.tensor([
            config.task.outcome_weights[key] for key in self._outcome_keys],
            **config.torchargs)

        self.inferrers = {
            outcome: Inferrer(1, config)
            for outcome in config.outkeys_o
        }

    def forward(self, actions: torch.Tensor, kstates: torch.Tensor, 
                states: torch.Tensor):
        _, batch_size, _ = states.shape
        outs = [self.inferrers[o].forward(actions, kstates, states)
                for o in self._outcome_keys]
        outs = u.safe.concat(outs, (batch_size, 1), dim=1, **self.torchargs)
        return outs  # batchsize * n_outcomes

    def q(self, qs: torch.Tensor, detach=True):
        q = qs @ self._outcome_weights
        return q.detach() if detach else q


class Actor(BaseNN):
    def __init__(self, config: NetConfig):
        super().__init__(config)

        self._total_state_size = sum(
            config.var(k).size for k in config.inkeys_a)
        
        self._total