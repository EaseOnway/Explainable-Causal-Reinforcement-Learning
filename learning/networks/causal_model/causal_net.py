from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from core import Batch, Buffer

from ..base import BaseNN
from .encoding import VariableEncoder
from .inferrer import Inferrer, StateKey
from learning.config import Config
import utils.tensorfuncs as T


class CausalNet(BaseNN):

    class Ablations:
        def __init__(self, no_attn=False, recur=False):
            self.no_attn = no_attn
            self.recur = recur

    def __init__(self, config: Config):
        super().__init__(config)

        dims = config.dims

        self.parent_dic: Dict[str, Tuple[str, ...]] = {}
        self.parent_dic_s: Dict[str, Tuple[str, ...]] = {}
        self.parent_dic_a: Dict[str, Tuple[str, ...]] = {}

        self.encoder = VariableEncoder(config)
        self.inferrers: Dict[str, Inferrer] = {}
        self.k_model = StateKey(config)

        for name in self.env.names_outputs:
            self.inferrers[name] = Inferrer(self.v(name).shape, config)
            self.add_module(f'{name}_inferrer', self.inferrers[name])

        # init parameters
        self.init_parameters()

    def load_graph(self, parent_dic: Dict[str, List[str]]):
        self.parent_dic.clear()
        self.parent_dic_s.clear()
        self.parent_dic_a.clear()

        for name in self.env.names_outputs:
            try:
                parents = parent_dic[name]
            except KeyError:
                parents = []

            self.parent_dic[name] = tuple(parents)
            self.parent_dic_s[name] = tuple(
                pa for pa in parents if pa in self.env.names_s)
            self.parent_dic_a[name] = tuple(
                pa for pa in parents if pa in self.env.names_a)

    def forward(self, datadic: Batch[torch.Tensor]):
        n = datadic.n  # batchsize

        encoded_data = self.encoder.forward_all(datadic)
        outs = Batch.torch(n)

        for var in self.env.names_outputs:
            inferrer = self.inferrers[var]
            parents_a = self.parent_dic_a[var]
            parents_s = self.parent_dic_s[var]
            actions_pa, k_states, states_pa = inferrer.input_from(\
                parents_a, parents_s, encoded_data, self.k_model)
            out = inferrer.forward(actions_pa, k_states, states_pa)
            outs[var] = out

        return outs

    def errors(self, datadic: Batch[torch.Tensor]):
        errors: Dict[str, torch.Tensor] = {}
        predicted = self.forward(datadic)
        for key, pred in predicted.items():
            inferrer = self.inferrers[key]
            target = datadic[key]
            errors[key] = inferrer.error(pred, target)
        return errors

    def loss(self, errors: Dict[str, torch.Tensor]):
        errs = [err for err in errors.values()]
        return torch.sum(torch.stack(errs))

    def get_attn_dic(self):
        out: Dict[str, torch.Tensor] = {}
        for var in self.env.names_outputs:
            attn = self.inferrers[var].attn  # nstates * batch
            attn = attn.detach().cpu().numpy()
            out[var] = attn
        return out
