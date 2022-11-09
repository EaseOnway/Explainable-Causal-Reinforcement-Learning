from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from ..data import Batch, Distributions, Transitions
from ..base import BaseNN
from .encoder import VariableEncoder
from .inferrer import StateKey, DistributionInferrer
from learning.config import Config
from utils.typings import NamedTensors, SortedParentDict, NamedArrays, NamedValues


class CausalNet(BaseNN):

    class Ablations:
        def __init__(self, no_attn=False, recur=False):
            self.no_attn = no_attn
            self.recur = recur

    def __init__(self, config: Config):
        super().__init__(config)

        dims = config.dims

        self.parent_dic: SortedParentDict = {}
        self.parent_dic_s: SortedParentDict = {}
        self.parent_dic_a: SortedParentDict = {}

        self.encoder = VariableEncoder(config)
        self.inferrers: Dict[str, DistributionInferrer] = {}
        self.k_model = StateKey(config)

        for name in self.env.names_outputs:
            self.inferrers[name] = DistributionInferrer(self.v(name), config)
            self.add_module(f'{name}_inferrer', self.inferrers[name])

        # init parameters
        self.init_parameters()

    def load_graph(self, parent_dic: Dict[str, Set[str]]):
        self.parent_dic.clear()
        self.parent_dic_s.clear()
        self.parent_dic_a.clear()

        for name in self.env.names_outputs:
            try:
                parents = parent_dic[name]
            except KeyError:
                parents = []

            self.parent_dic[name] = tuple(sorted(parents))
            self.parent_dic_s[name] = tuple(sorted(
                pa for pa in parents if pa in self.env.names_s))
            self.parent_dic_a[name] = tuple(sorted(
                pa for pa in parents if pa in self.env.names_a))

    def forward(self, raw_data: Batch) -> Distributions:
        data = raw_data.kapply(self.raw2input)
        encoded_data = self.encoder.forward_all(data)
        outs = Distributions(data.n)

        for var in self.env.names_outputs:
            inferrer = self.inferrers[var]
            parents_a = self.parent_dic_a[var]
            parents_s = self.parent_dic_s[var]
            actions_pa, k_states, states_pa = inferrer.input_from(\
                parents_a, parents_s, encoded_data, self.k_model)
            out = inferrer.forward(actions_pa, k_states, states_pa)
            outs[var] = out

        return outs

    def get_loglikeli_dic(self, raw_data: Batch):
        '''get the dictionary of log-likelihoood'''
        
        labels = raw_data.select(self.env.names_outputs).kapply(self.raw2label)
        predicted = self.forward(raw_data)
        logprobs = predicted.logprobs(labels)
        return {k: torch.mean(v) for k, v in logprobs.items()}

    def loglikelihood(self, lldic: NamedTensors):
        ls = list(lldic.values())
        return torch.sum(torch.stack(ls))

    def get_attn_dic(self):
        out: NamedTensors = {}
        for var in self.env.names_outputs:
            attn = self.inferrers[var].attn  # nstates * batch
            attn = attn.detach().cpu().numpy()
            out[var] = attn
        return out
    
    def simulate(self, state_actions: NamedValues):
        data =  Batch.from_sample(self.as_raws(state_actions))
        dis = self.forward(data)
        out = dis.sample().kapply(self.label2raw)
        out = self.as_numpy(out, drop_batch=True)
        return out