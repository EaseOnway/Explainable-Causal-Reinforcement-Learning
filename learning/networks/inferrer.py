from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

from .config import NetConfig
from .encoding import Aggregator
from .base import BaseNN
import utils as u


class StateKey(BaseNN):
    def __init__(self, config: NetConfig):
        super().__init__(config)

        self.K = nn.parameter.Parameter(
            torch.zeros((len(config.inkeys_s), config.dims.k),
                        **config.torchargs)
        )

        self.__indexmap = {s: i for i, s in enumerate(config.inkeys_s)}

    def forward(self, state_names: Tuple[str]):
        i = tuple(self.__indexmap[name] for name in state_names)

        return self.K[i, :]


class Inferrer(BaseNN):
    def __init__(self, shape_out: Tuple[int, ...], config: NetConfig,
                 categorical=False):
        super().__init__(config)
        self._shape_out = shape_out
        self._size_out = u.get_size(shape_out)
        self._categorical = categorical

        dims = self.dims
        if config.ablations.recur:
            d = max(dims.a, dims.s)
            self.aggregator = Aggregator(d, dims.v, config)
        else:
            self.aggregator = Aggregator(dims.a, dims.a_emb, config)
            self.linear_q = nn.Linear(dims.a_emb, dims.k, **config.torchargs)
            self.linear_vs = nn.Linear(dims.s, dims.v, **config.torchargs)
            self.linear_va = nn.Linear(dims.a_emb, dims.v, **config.torchargs)

        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dims.v, dims.h_dec, **config.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dims.h_dec, dims.h_dec, **config.torchargs),
            nn.PReLU(dims.h_dec, **config.torchargs),
            nn.Linear(dims.h_dec, self._size_out, **config.torchargs)
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
        da, ds = self.dims.a, self.dims.s

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
        q = self.linear_q(emb_a)  # batch * dim_k
        scores: torch.Tensor = torch.sum(
            kstates * q, dim=2) / np.sqrt(self.dims.k)  # num_state * batch
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

    def forward(self, actions: torch.Tensor, kstates: torch.Tensor,
                states: torch.Tensor):
        if self.ablations.recur:
            x = self.__rec_infer(actions, kstates, states)
        else:
            x = self.__attn_infer(actions, kstates, states)

        out: torch.Tensor = self.decoder(x)  # batchsize * dim_out
        if self._categorical:
            out = torch.softmax(out, dim=1)
        return out.view(out.shape[0], *self._shape_out)
    
    def error(self, out: torch.Tensor, target: np.ndarray):
        if self._categorical:
            prob = torch.softmax(out, dim=1)
            indices = u.transform.onehot_indices(target)
            e = -(torch.log(prob[indices] + 1e-20)) + 1e-20
            return torch.mean(e)
        else:
            target_ = torch.from_numpy(target).to(**self.torchargs)
            return torch.mean(torch.square(target_ - out))
    
    def predict(self, out: torch.Tensor) -> np.ndarray:
        out = out.detach()
        if self._categorical:
            pred = u.reduction.batch_argmax(out)
            pred = pred.cpu().numpy()
        else:
            pred = out.cpu().numpy()
        return pred
