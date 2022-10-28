from typing import Sequence, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from core.data import Batch

from learning.config import Config
from .encoding import Aggregator
from ..base import BaseNN
import utils.tensorfuncs as T
import utils.shaping as shaping


class StateKey(BaseNN):
    def __init__(self, config: Config):
        super().__init__(config)
        self.K = nn.parameter.Parameter(
            torch.zeros(self.env.num_s, self.dims.inferrer_key),
                        **self.torchargs)

    def forward(self, state_names: Sequence[str]):
        i = tuple(self.env.idx_s(name) for name in state_names)

        return self.K[i, :]


class Inferrer(BaseNN):
    def __init__(self, shape_out: Tuple[int, ...], config: Config):
        super().__init__(config)
        self._shape_out = shape_out
        self._size_out = shaping.get_size(shape_out)

        da, ds = self.dims.action_encoding, self.dims.state_encoding
        dv, dk = self.dims.inferrer_value, self.dims.inferrer_key
        dh_agg = self.dims.action_aggregator_hidden
        dh_dec = self.dims.inferrer_decoder_hidden

        if config.ablations.recur:
            d = max(da, ds)
            self.aggregator = Aggregator(d, dv, dh_agg, config)
        else:
            self.aggregator = Aggregator(da, dk, dh_agg, config)
            self.linear_vs = nn.Linear(ds, dv, **self.torchargs)
            self.linear_va = nn.Linear(dk, dv, **self.torchargs)

        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dv, dh_dec, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dh_dec, dh_dec, **self.torchargs),
            nn.PReLU(dh_dec, **self.torchargs),
            nn.Linear(dh_dec, self._size_out, **self.torchargs)
        )

        self.attn: torch.Tensor

    def __rec_infer(self,
                    # parent_actions * batch * dim_a
                    actions: torch.Tensor,
                    # parent_states * dim_k
                    kstates: torch.Tensor,
                    # parent_states * batch * dim_s
                    states: torch.Tensor,
                    ):
        
        da, ds = self.dims.action_encoding, self.dims.state_encoding
        batchsize = actions.shape[1]
        assert states.shape[1] == batchsize

        d = max(da, ds)
        if da < d:
            n = actions.shape[0]
            actions = torch.concat([
                actions,
                torch.zeros(n, batchsize, d - da,
                            requires_grad=False, **self.torchargs)
            ], dim=2)
        if ds < d:
            n = states.shape[0]
            states = torch.concat([
                states,
                torch.zeros(n, batchsize, d - da,
                            requires_grad=False, **self.torchargs)
            ], dim=2)

        seq = torch.concat((states, actions), dim=0)
        out = self.aggregator.forward(seq)

        return out

    def get_attn_scores(self, emb_a: torch.Tensor, kstates: torch.Tensor):
        num_state, _ = kstates.shape
        kstates = kstates.unsqueeze(dim=1)  # num_state * 1 *dim_k
        q = emb_a  # batch * dim_k
        scores: torch.Tensor = torch.sum(
            kstates * q, dim=2) / np.sqrt(self.dims.inferrer_key)  # num_state * batch
        attn = torch.softmax(scores, dim=0)  # num_state * batch
        return attn

    def attn_infer(self, attn: torch.Tensor,
                   emb_a: torch.Tensor, states: torch.Tensor):
        # emb_a: batch * dim_emb_a
        # attn: num_states * batch
        # states: num_states * batch * dim_s

        num_state, batch_size = states.shape[:2]
        assert emb_a.shape[0] == batch_size

        vs = self.linear_vs(states)  # num_states * batch * dim_v
        vs = torch.sum(vs * attn.view(num_state, batch_size, 1),
                       dim=0)  # batch * dim_v
        va = self.linear_va(emb_a)   # batch * dim_v
        out: torch.Tensor = vs + va  # batch * dim_v

        # out: batch * dim_v
        return out

    def __attn_infer(self,
                     actions: torch.Tensor,
                     kstates: torch.Tensor,
                     states: torch.Tensor):
        # actions: num_actions * batch * dim_a
        # kstates: num_states * dim_k
        # states: num_states * batch * dim_s
        emb_a = self.aggregator(actions)
        if not self.ablations.no_attn:
            attn = self.get_attn_scores(emb_a, kstates)  # num_states * batch
        else:
            num_state, batch_size = states.shape[:2]
            attn = torch.full((num_state, batch_size), 1/num_state,
                              **self.torchargs)

        out = self.attn_infer(attn, emb_a, states)
        self.attn = attn
        return out

    @staticmethod
    def input_from(action_keys: Sequence[str], state_keys: Sequence[str],
                   data: Batch[torch.Tensor], key_model: StateKey):
        actions = T.safe_stack([data[k] for k in action_keys],
                               (data.n, key_model.dims.action_encoding),
                                **key_model.torchargs)
        states = T.safe_stack([data[k] for k in sorted(state_keys)],
                              (data.n, key_model.dims.state_encoding),
                               **key_model.torchargs)
        kstates = key_model.forward(state_keys)
        return actions, kstates, states

    def forward(self, actions: torch.Tensor, kstates: torch.Tensor,
                states: torch.Tensor):
        if self.ablations.recur:
            x = self.__rec_infer(actions, kstates, states)
        else:
            x = self.__attn_infer(actions, kstates, states)

        out: torch.Tensor = self.decoder(x)  # batchsize * dim_out
        return out.view(out.shape[0], *self._shape_out)
    
    def error(self, out: torch.Tensor, target: torch.Tensor):
        return torch.mean(torch.square(target - out))
    
    def predict(self, out: torch.Tensor) -> np.ndarray:
        return T.t2a(out)
