from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import random

from core import Batch, Distributions, Transitions
from ..base import BaseNN, Context, RLBase
from .modules import StateKey, DistributionInferrer, DistributionDecoder, \
    VariableEncoder, VariableConcat
from utils.typings import NamedTensors, ParentDict, NamedArrays, NamedValues


class EnvModel(BaseNN):
    def __init__(self, context: Context):
        super().__init__(context)

    def forward(self, raw_data: Batch) -> Distributions:
        raise NotImplementedError
    
    def get_loglikeli_dic(self, raw_data: Batch):
        '''get the dictionary of log-likelihoood'''
        
        labels = raw_data.select(self.env.names_output).kapply(self.raw2label)
        predicted = self.forward(raw_data)
        logprobs = predicted.logprobs(labels)
        return {k: torch.mean(v) for k, v in logprobs.items()}

    def loglikelihood(self, lldic: NamedTensors):
        ls = list(lldic.values())
        return torch.sum(torch.stack(ls))
    
    def simulate(self, state_actions: NamedValues, mode=False):
        self.train(False)
        with torch.no_grad():
            data =  Batch.from_sample(self.named_tensors(state_actions))
            dis = self.forward(data)
            if mode:
                out = dis.mode()
            else:
                out = dis.sample()
            out = out.kapply(self.label2raw)
            out = self.as_numpy(out, drop_batch=True)
        return out

    def optimizer(self):
        return self.F.get_optmizer(self.config.model.optim, self)


class CausalEnvModel(EnvModel):

    def __init__(self, context: Context):
        super().__init__(context)

        self.parent_dic: ParentDict = {}
        self.parent_dic_s: ParentDict = {}
        self.parent_dic_a: ParentDict = {}

        self.encoder = VariableEncoder(context)
        self.inferrers: Dict[str, DistributionInferrer] = {}
        self.k_model = StateKey(context)

        for name in self.env.names_output:
            self.inferrers[name] = DistributionInferrer(self.v(name), context)
            self.add_module(f'{name}_inferrer', self.inferrers[name])
    
    def load_graph(self, parent_dic: ParentDict):
        self.parent_dic.clear()
        self.parent_dic_s.clear()
        self.parent_dic_a.clear()

        for name in self.env.names_output:
            try:
                parents = set(parent_dic[name])
            except KeyError:
                parents = set()

            self.parent_dic[name] = tuple(sorted(parents))
            self.parent_dic_s[name] = tuple(sorted(
                pa for pa in parents if pa in self.env.names_s))
            self.parent_dic_a[name] = tuple(sorted(
                pa for pa in parents if pa in self.env.names_a))

    def forward(self, raw_data: Batch) -> Distributions:
        data = raw_data.kapply(self.raw2input)
        encoded_data = self.encoder.forward(data)
        outs = Distributions(data.n)

        for var in self.env.names_output:
            inferrer = self.inferrers[var]
            parents_a = self.parent_dic_a[var]
            parents_s = self.parent_dic_s[var]
            actions_pa, k_states, states_pa = inferrer.input_from(\
                parents_a, parents_s, encoded_data, self.k_model)
            out = inferrer.forward(actions_pa, k_states, states_pa)
            outs[var] = out

        return outs


class MLPEnvModel(EnvModel):

    def __init__(self, context: Context):
        super().__init__(context)

        self.encoder = VariableConcat(context, self.env.names_input)
        dim = self.dims.mlp_model_hidden
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.size, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
        )
        self.decoders: Dict[str, DistributionDecoder] = {}
        self.k_model = StateKey(context)

        for name in self.env.names_output:
            self.decoders[name] = DistributionDecoder(dim, self.v(name), context)
            self.add_module(f'{name}_decoder', self.decoders[name])

    def forward(self, raw_data: Batch) -> Distributions:
        concat_data = self.encoder.forward(raw_data)
        encoded_data: torch.Tensor = self.mlp(concat_data)
        outs = Distributions(raw_data.n)

        for var in self.env.names_output:
            decoder = self.decoders[var]
            out = decoder.forward(encoded_data)
            outs[var] = out

        return outs


class EnvModelEnsemble(EnvModel):
    def __init__(self, context: Context, networks: Tuple[EnvModel, ...]):
        super().__init__(context)

        self.networks = networks
        for i, network in enumerate(self.networks):
            if isinstance(network, EnvModelEnsemble):
                raise TypeError("we do not accept an ensemble as a sub-network")
            self.add_module(f'model_{i}', network)

    def optimizers(self):
        return tuple(net.optimizer() for net in self.networks)

    def __getitem__(self, i: int):
        return self.networks[i]
    
    def __len__(self):
        return len(self.networks)
    
    def __iter__(self):
        return iter(self.networks)

    def load_graph(self, parent_dic: ParentDict):
        for network in self.networks:
            if isinstance(network, CausalEnvModel):
                network.load_graph(parent_dic)

    def random_select(self):
        i = random.randrange(len(self.networks))
        return i, self.networks[i]
    
    def forward(self, raw_data: Batch) -> Distributions:
        _, net = self.random_select()
        return net.forward(raw_data)
