from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from core import TaskInfo, VarInfo

from .base import BaseNN
from .encoding import VariableEncoder
from .inferrer import Inferrer, StateKey
from .config import *
from .utils import safe_stack


class CausalNet(BaseNN):

    def __init__(self, config: NetConfig):
        super().__init__(config)

        dims = config.dims

        self.parent_dic: Dict[str, Tuple[str, ...]] = {}
        self.parent_dic_s: Dict[str, Tuple[str, ...]] = {}
        self.parent_dic_a: Dict[str, Tuple[str, ...]] = {}

        self.action_encoder = VariableEncoder(
            dims.a, list(config.task.action_keys), config)
        self.state_encoder = VariableEncoder(
            dims.s, list(config.task.in_state_keys), config)
        self.inferrers: Dict[str, Inferrer] = {}
        self.k_model = StateKey(config)

        for key in config.outkeys:
            self.inferrers[key] = Inferrer(self.config.var(key).size, config)
            self.add_module(f'{key}_inferrer', self.inferrers[key])

        # init parameters
        for p in self.parameters():
            if p.ndim < 2:
                nn.init.normal_(p)
            else:
                nn.init.xavier_normal_(p)

    def load_graph(self, parent_dic: Dict[str, List[str]]):
        self.parent_dic.clear()
        self.parent_dic_s.clear()
        self.parent_dic_a.clear()

        for var in self.config.outkeys:
            try:
                parents = parent_dic[var]
            except KeyError:
                parents = []

            self.parent_dic[var] = tuple(parents)
            self.parent_dic_s[var] = tuple(
                pa for pa in parents if pa in self.config.inkeys_s)
            self.parent_dic_a[var] = tuple(
                pa for pa in parents if pa in self.config.inkeys_a)

    def forward(self, datadic: Dict[str, np.ndarray]):
        try:
            batchsize = next(iter(datadic.values())).shape[0]
        except StopIteration:
            raise ValueError("data is empty")

        actions = self.action_encoder.forward(datadic)
        states = self.state_encoder.forward(datadic)

        outs: Dict[str, torch.Tensor] = {}

        for var in self.config.outkeys:
            parents_a = self.parent_dic_a[var]
            parents_s = self.parent_dic_s[var]
            actions_pa = safe_stack([actions[pa] for pa in parents_a],
                                    (batchsize, self.dims.a), **self.torchargs)
            states_pa = safe_stack([states[pa] for pa in parents_s],
                                   (batchsize, self.dims.s), **self.torchargs)
            k_states = self.k_model.forward(parents_s)
            inferrer = self.inferrers[var]
            out = inferrer.forward(actions_pa, k_states, states_pa)
            outs[var] = out

        return outs

    def errors(self, data_dic: Dict[str, np.ndarray]):
        errors: Dict[str, torch.Tensor] = {}

        predicted = self.forward(data_dic)
        for key, pred in predicted.items():
            target = data_dic[key]
            target = torch.from_numpy(target).to(
                **self.torchargs).view(target.shape[0], -1)
            errors[key] = torch.mean(torch.square(target - pred))
        return errors

    def loss(self, errors: Dict[str, torch.Tensor]):
        s = 0
        errs = []
        for key, err in errors.items():
            size = self.config.var(key).size
            s += size
            errs.append(err * size)
        return torch.sum(torch.stack(errs))/s

    def get_attn_dic(self):
        out: Dict[str, np.ndarray] = {}
        for var in self.config.outkeys:
            attn = self.inferrers[var].attn  # nstates * batch
            attn = attn.detach().cpu().numpy()
            out[var] = attn
        return out
