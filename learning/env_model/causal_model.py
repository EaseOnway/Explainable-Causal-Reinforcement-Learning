from typing import List, Sequence

import torch
import torch.nn as nn
import numpy as np

from .env_model import EnvModel
from ..base import BaseNN, Context
from utils.typings import SortedNames, ParentDict
from utils import MultiLinear
from core import Batch, VType, Distributions
from .modules import VariableEncoder, DistributionDecoder, Aggregator


class CausalModel(EnvModel):
    def __init__(self, context):
        super().__init__(context)

        self._idx_input = {name: i for i, name in enumerate(self.env.names_input)}
        self._idx_output = {name: i for i, name in enumerate(self.env.names_output)}
        self._idx_s = {name: i for i, name in enumerate(self.env.names_s)}
        self._idx_a = {name: i for i, name in enumerate(self.env.names_a)}
        self._action_indices = tuple(self._idx_input[name] for name in self.env.names_a)
        self._state_indices = tuple(self._idx_input[name] for name in self.env.names_s)

        self.mask_input: torch.Tensor
        self.mask_action: torch.Tensor
        self.mask_state: torch.Tensor
        self.register_buffer("mask_input", torch.zeros(
            self.env.num_output, self.env.num_input,
            dtype=torch.bool, device=self.device))
        self.register_buffer("mask_action", torch.zeros(
            self.env.num_output, self.env.num_a,
            dtype=torch.bool, device=self.device))
        self.register_buffer("mask_state", torch.zeros(
            self.env.num_output, self.env.num_s,
            dtype=torch.bool, device=self.device))
        
        self.decoders = [
            DistributionDecoder(self.dims.distribution_embedding, self.v(name), context)
            for name in self.env.names_output
        ]
        for i, decoder in enumerate(self.decoders):
            self.add_module(f'{self.env.names_output[i]}_decoder', decoder)

    
    def load_graph(self, g: ParentDict):
        for name_out, parents in g.items():
            j = self._idx_output[name_out]
            for name_in in parents:
                i = self._idx_input[name_in]
                self.mask_input[j, i] = True
                try:
                    i = self._idx_a[name_in]
                except KeyError:
                    pass
                else:
                    self.mask_action[j, i] = True
                try:
                    i = self._idx_s[name_in]
                except KeyError:
                    pass
                else:
                    self.mask_state[j, i] = True
        
    @property
    def causal_graph(self) -> ParentDict:
        return {
            self.env.names_output[j]: tuple(
                self.env.names_input[i]
                for i in range(self.env.num_input)
                if self.mask_input[j, i] == True
            ) for j in range(self.env.num_output)
        }

    def infer(self, input_: Batch) -> torch.Tensor:
        '''
        return: distribution embedding (batch, num_output, dim_embedding)
        '''
        raise NotImplementedError

    def forward(self, raw_data: Batch) -> Distributions:
        input_ = raw_data.kapply(self.raw2input)
        distribution_embedding = self.infer(input_)
        out = Distributions(raw_data.n)
        for i, name in enumerate(self.env.names_output):
            dis = self.decoders[i].forward(distribution_embedding[:, i])
            out[name] = dis
        return out


class AttnCausalModel(CausalModel):
    def __init__(self, context):
        super().__init__(context)

        self.encoder = VariableEncoder(context, self.env.names_input)
        dims = self.dims

        self.keys = nn.Parameter(torch.randn(
            self.env.num_input, dims.inferrer_key,
            **self.torchargs
        ))
        self.query_net = MultiLinear(
             [self.env.num_output],
             dims.variable_encoding * self.env.num_a, dims.inferrer_key,
             **self.torchargs
        )
        self.value_net = MultiLinear(
            [self.env.num_output, 1],
            dims.variable_encoding, dims.inferrer_value,
            **self.torchargs
        )
        self.feed_forward = MultiLinear(
             [self.env.num_output],
             dims.inferrer_value, dims.distribution_embedding,
             **self.torchargs
        )
    
    def compute_query(self, encoding: torch.Tensor):
        # fetch action encoding
        N = encoding.shape[0]
        num_a = self.env.num_a
        num_q = self.env.num_output

        encoding = encoding[:, self._action_indices, :]  # batch * num_a * de
        encoding = encoding.unsqueeze(1)  # batch * 1 * num_a * de
        encoding = encoding.expand(-1, num_q, -1, -1)  # batch * num_q * num_a * de

        # apply mask
        mask = self.mask_action  # num_q * num_a
        mask = mask.unsqueeze(-1)  # num_q * num_a * 1
        encoding = encoding * mask  # batch * num_q * num_a * de
    
        # apply transform
        encoding = encoding.reshape(
            N, num_q, (num_a * self.dims.variable_encoding))  # batch * num_q * (num_a * dk)
        query = self.query_net.forward(encoding)  # batch * num_q * dk

        return query
    
    def infer(self, input_: Batch) -> torch.Tensor:
        encoding = self.encoder.forward(input_)  # batch * num_in * de
        q = self.compute_query(encoding)  # batch * num_q * dk
        k = self.keys  # num_in * dk
        
        v = self.value_net.forward(encoding.unsqueeze(1))  # batch * num_q * num_in * dv
        
        s = torch.matmul(q, k.T) / np.sqrt(self.dims.inferrer_key) # batch * num_q * num_in
        s = torch.masked_fill(s, torch.logical_not(self.mask_input), -np.inf)
        exps = torch.exp(s)  # batch * num_q * num_in
        attn = exps / (torch.sum(exps, dim=2) + 1.0)
        # batch * num_q * num_in

        self.weight = attn
        
        x = v * attn.unsqueeze(-1)  # batch * num_q * num_in * dv
        x = x.reshape.sum(dim=2)   # batch * num_q * dv

        y = self.feed_forward.forward(x)
        return y


class RecurrentCausalModel(CausalModel):
    def __init__(self, context: Context):
        super().__init__(context)
        
        de = self.dims.variable_encoding
        dv = self.dims.inferrer_value
        dh_agg = self.dims.aggregator_hidden
        dims = self.dims

        self.encoder = VariableEncoder(context, self.env.names_input)

        self.inferrers = [  # recurrent aggregators
            Aggregator(de, dims.distribution_embedding, dh_agg, context)
            for name in self.env.names_output
        ]
        for i, aggregator in enumerate(self.inferrers):
            self.add_module(f'{self.env.names_output[i]}_aggregator', aggregator)


    def infer(self, input_: Batch):
        encoding = self.encoder.forward(input_)
        temp = []

        for i, aggregator in enumerate(self.inferrers):
            mask = self.mask_input[i, :]
            seq = encoding[:, mask, :]
            seq = seq.transpose(1, 0)
            out = aggregator.forward(seq)
            temp.append(out)

        distribution_embedding = self.T.safe_stack(temp, (input_.n, self.dims.distribution_embedding))
        distribution_embedding = distribution_embedding.transpose(0, 1)
        return distribution_embedding
