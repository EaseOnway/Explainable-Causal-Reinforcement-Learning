from typing import Sequence, Tuple, Optional
import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from utils import Shaping
from core import Batch
from learning.config import Config
from .encoder import Aggregator
from ..base import BaseNN
from core.vtype import VType


class StateKey(BaseNN):
    def __init__(self, config: Config):
        super().__init__(config)
        self.idx_map = {k: i for i, k in enumerate(self.env.names_s)}
        self.K = nn.parameter.Parameter(
            torch.zeros(self.env.num_s, self.dims.inferrer_key,
                        **self.torchargs))

    def forward(self, state_names: Sequence[str]):
        i = tuple(self.idx_map[name] for name in state_names)
        return self.K[i, :]


class Inferrer(BaseNN):
    def __init__(self, config: Config):
        super().__init__(config)

        d = self.dims.variable_encoding
        d_emb = self.dims.action_influce_embedding
        dv, dk = self.dims.inferrer_value, self.dims.inferrer_key
        dh_agg = self.dims.aggregator_hidden
        dff = self.dims.inferrer_feed_forward

        if config.ablations.recur:
            self.aggregator = Aggregator(d, dv, dh_agg, config)
        else:
            self.aggregator = Aggregator(d, d_emb, dh_agg, config)
            self.linear_vs = nn.Linear(d, dv, **self.torchargs)
            self.linear_va = nn.Linear(d_emb, dv, **self.torchargs)
            self.linear_q = nn.Linear(d_emb, dk, **self.torchargs)
            self.layernorm = nn.LayerNorm([dv], **self.torchargs,
                                          elementwise_affine=False)

        self.feed_forward = nn.Sequential(
            nn.Linear(dv, dff, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dff, dff, **self.torchargs),
            nn.LeakyReLU(),
        )

        self.attn: Tuple[torch.Tensor, torch.Tensor]

    def __rec_infer(self,
                    # parent_actions * batch * dim_a
                    actions: torch.Tensor,
                    # parent_states * dim_k
                    kstates: torch.Tensor,
                    # parent_states * batch * dim_s
                    states: torch.Tensor,
                    ):

        batchsize = actions.shape[1]
        assert states.shape[1] == batchsize

        d = self.dims.variable_encoding
        seq = torch.concat((states, actions), dim=0)
        out = self.aggregator.forward(seq)

        return out

    def get_attn_scores(self, emb_a: torch.Tensor, kstates: torch.Tensor):
        num_state, _ = kstates.shape
        kstates = kstates.unsqueeze(dim=1)  # num_state * 1 *dim_k
        q = self.linear_q(emb_a)  # batch * dim_k
        scores: torch.Tensor = torch.sum(
            kstates * q, dim=2) / np.sqrt(self.dims.inferrer_key)  # num_state * batch
        expscores = torch.exp(scores)
        sumexpscores = torch.sum(expscores, dim=0) + 1  # batch
        attn_s = expscores / sumexpscores # attn_s: num_states * batch
        attn_a = 1 / sumexpscores  # attn_a: batch
        return attn_s, attn_a

    def attn_infer(self, attn_s: torch.Tensor, attn_a: torch.Tensor,
                   emb_a: torch.Tensor, states: torch.Tensor):
        # emb_a: batch * dim_emb_a
        # attn_s: num_states * batch
        # attn_a: batch
        # states: num_states * batch * dim_s

        num_state, batch_size = states.shape[:2]
        assert emb_a.shape[0] == batch_size

        vs: torch.Tensor = self.linear_vs(states)  # num_states * batch * dim_v
        va: torch.Tensor = self.linear_va(emb_a)   # batch * dim_v

        v = torch.cat((vs, va.unsqueeze(0)), dim=0)  # (num_states + 1) * batch * dim_v
        v: torch.Tensor = self.layernorm(v)
        a = torch.cat((attn_s, attn_a.unsqueeze(0)), dim=0)  # (num_states + 1) * batch
        a = a.view((num_state + 1), batch_size, 1)
          
        return torch.sum(v * a, dim=0)  # batch * dim_v

    def __attn_infer(self, actions: torch.Tensor, kstates: torch.Tensor,
                     states: torch.Tensor):
        # actions: num_actions * batch * dim_a
        # kstates: num_states * dim_k
        # states: num_states * batch * dim_s
        emb_a = self.aggregator(actions)
        if not self.ablations.no_attn:
            attn_s, attn_a = self.get_attn_scores(emb_a, kstates)  # num_states * batch
        else:
            num_state, batch_size = states.shape[:2]
            attn = 1/(num_state + 1)
            attn_s = torch.full((num_state, batch_size), attn, **self.torchargs)
            attn_a = torch.full((batch_size,), attn, **self.torchargs)

        out = self.attn_infer(attn_s, attn_a, emb_a, states)
        self.attn = (attn_s, attn_a)
        return out

    def input_from(self, action_keys: Sequence[str], state_keys: Sequence[str],
                   encoded_data: Batch, key_model: StateKey):
        T = self.T
        dims = self.dims
        actions = T.safe_stack([encoded_data[k] for k in action_keys],
                               (encoded_data.n, dims.variable_encoding))
        states = T.safe_stack([encoded_data[k] for k in sorted(state_keys)],
                              (encoded_data.n, dims.variable_encoding))
        kstates = key_model.forward(state_keys)
        return actions, kstates, states

    def forward(self, actions: torch.Tensor, kstates: torch.Tensor,
                states: torch.Tensor):
        if self.ablations.recur:
            x = self.__rec_infer(actions, kstates, states)
        else:
            x = self.__attn_infer(actions, kstates, states)

        x = self.feed_forward(x)
        return x


class DistributionDecoder(BaseNN):
    def __init__(self, dim_in: int, vtype: VType, config: Config):
        super().__init__(config)
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


class DistributionInferrer(Inferrer):
    def __init__(self, vtype: VType, config: Config):
        super().__init__(config)
        dff = config.dims.inferrer_feed_forward
        self.decoder = DistributionDecoder(dff, vtype, config)
    
    def forward(self, actions: torch.Tensor, kstates: torch.Tensor,
                states: torch.Tensor):
        x = super().forward(actions, kstates, states)
        out = self.decoder.forward(x)
        return out
