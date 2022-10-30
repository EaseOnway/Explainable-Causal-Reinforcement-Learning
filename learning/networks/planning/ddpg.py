from __future__ import annotations

from typing import Dict, Iterable, Optional, Type, final
import torch
import torch.nn as nn
import numpy as np

from ..causal_model import CausalNet
from ..causal_model.inferrer import Inferrer
from ..base import BaseNN
from learning.config import Configured, Config
import utils.tensorfuncs as T
from core import Batch, Buffer


class Critic(BaseNN):
    def __init__(self, config: Config):
        super().__init__(config)

        self.inferrers = {outcome: Inferrer((), config)
                          for outcome in self.env.names_o}

        for k, inferrer in self.inferrers.items():
            self.add_module(f"Q({k}) inferrer ", inferrer)

    def forward(self, actions: torch.Tensor, kstates: torch.Tensor, 
                states: torch.Tensor):
        _, batch_size, _ = states.shape
        outs = [self.inferrers[o].forward(actions, kstates, states)
                for o in self.env.names_o]
        outs = T.batch_flatcat(outs, batch_size, **self.torchargs)
        return outs  # batchsize * n_outcomes

    def q(self, qs: torch.Tensor, detach=True):
        q = qs @ self.get_outcome_weights()
        if detach:
            q = q.detach_()
        return q
        

class ActorEncoder(BaseNN):
    def __init__(self, config: Config,
                 restriction: Optional[Iterable[str]] = None):
        super().__init__(config)

        if restriction is None:
            self._pa = self.env.names_s
        else:
            self._pa = sorted(set(restriction))
        
        self._in_dim = sum(self.v(k).size for k in self._pa)

        dim = config.dims.actor_hidden
        self.__net = nn.Sequential( 
            nn.Linear(self._in_dim, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
        )
    
    def forward(self, states: Batch[torch.Tensor]):
        to_cat = [states[pa] for pa in self._pa]
        x = T.batch_flatcat(to_cat, states.n, **self.torchargs)
        e = self.__net(x)
        return e


class Actor(BaseNN):
    def __init__(self, config: Config):
        super().__init__(config)

        self._akeys = self.env.names_a
        
        self.encoder = ActorEncoder(config)

        self.decoders = {var: self.__make_decoder(var)
                         for var in self._akeys}

    def __make_decoder(self, var: str) -> nn.Module:
        linear = nn.Linear(self.dims.actor_hidden,
                           self.v(var).size, **self.torchargs)
        self.add_module(f'{var} decoder', linear)
        return linear
    
    def forward(self, states: Batch[torch.Tensor],
                explore_sd: Optional[float] = None):
        e = self.encoder.forward(states)
        
        out = Batch.torch(states.n)
        for k, d in self.decoders.items():
            a = d(e).view(states.n, *self.v(k).shape)
            if explore_sd is not None:
                a += torch.randn_like(a) * explore_sd
            out[k] = a
        
        return out


class DDPG(Configured):
    def __init__(self, config: Config, causnet: CausalNet):
        super().__init__(config)
        self.args = config.ddpg_args

        self.actor = Actor(config)
        self.critic = Critic(config)
        self.target_actor = Actor(config)
        self.target_critic = Critic(config)
        self.causnet = causnet
        self.actor.init_parameters()
        self.critic.init_parameters()
        self.update_target_networks(hard=True)

        self.opt_a = self.config.train_args.get_optimizer(self.actor)
        self.opt_c = self.config.train_args.get_optimizer(self.critic)

    def get_inputs_for_critic(self, states_actions: Batch[torch.Tensor]):
        data = self.causnet.encoder.forward_all(states_actions)
        actions, kstates, states = Inferrer.input_from(
            self.env.names_a, self.env.names_s, data, self.causnet.k_model)
        kstates.detach_()
        states.detach_()
        return actions, kstates, states

    def actor_loss(self, data: Batch[torch.Tensor]):
        data = data.select(self.env.names_s)
        a = self.actor.forward(data)
        data.update(a)
        qs = self.critic.forward(*self.get_inputs_for_critic(data))
        q = self.critic.q(qs, detach=False)
        negmeanq = - torch.mean(q)  # negative gradient for policy ascent
        return negmeanq
    
    def __td_targets(self, data: Batch[torch.Tensor]):
        # 计算reward
        outcomes = self.get_outcome_vectors(data)

        # 构造next data
        next_data = self.step_shift(data)

        with torch.no_grad():
            next_a = self.target_actor.forward(next_data)
            next_data.update(next_a)
            critic_inputs = self.get_inputs_for_critic(next_data)
            next_qs = self.target_critic.forward(*critic_inputs)
            out = next_qs * self.args.gamma + outcomes
        
        return out
    
    def critic_loss(self,  data: Batch[torch.Tensor]):
        qs = self.critic.forward(*self.get_inputs_for_critic(data))
        targets = self.__td_targets(data)
        targets.detach_()
        return torch.mean(torch.square(qs - targets))
    
    def update_target_networks(self, hard=False):
        rate = 1.0 if hard else self.args.target_update_rate

        def update(model: nn.Module, target: nn.Module):
            for name, p_target in target.named_parameters():
                p = model.get_parameter(name)
                p_target.data = p_target.data * (1 - rate) + p.data * rate
        
        update(self.actor, self.target_actor)
        update(self.critic, self.target_critic)

    def train_batch(self, data: Batch[torch.Tensor]):
        self.opt_c.zero_grad()
        loss_c = self.critic_loss(data)
        loss_c.backward()
        self.opt_c.step()

        self.opt_a.zero_grad()
        loss_a = self.actor_loss(data)
        loss_a.backward()
        self.opt_a.step()

        self.update_target_networks(hard=False)

        return float(loss_a), float(loss_c)

    def dream(self, buffer: Buffer):
        data_np = buffer.sample_batch(self.config.train_args.batchsize)
        data = self.a2t(data_np)

        # choose a subset to dream

        data = data.select(self.env.names_inputs)
        with torch.no_grad():
            data.update(self.causnet.forward(data))

        return data
