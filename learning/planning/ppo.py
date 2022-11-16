from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Any

import torch
import torch.nn as nn

from ..causal_model.inferrer import DistributionDecoder
from ..buffer import Buffer
from ..config import Config
from ..base import Configured, BaseNN
from ..data import Batch, Transitions, Distributions
from utils.typings import NamedValues
from core import DType
import utils as u


_ADV = "_ADV_"  # key for advantage
_TARGET = "_TARGET_"  # key for td-target



class VariableConcat(BaseNN):
    def __init__(self, config: Config, var_names: Sequence[str]):
        super().__init__(config)

        self.__names = tuple(var_names)
        self.__size = sum(self.v(k).size for k in self.__names)

    @property
    def names(self):
        return self.__names

    @property
    def size(self):
        return self.__size

    def forward(self, raw: Batch):
        to_cat = [self.raw2input(name, raw[name]) for name in self.__names]
        x = self.T.safe_cat(to_cat, (raw.n, -1), 1)
        assert x.shape[1] == self.__size
        return x


class StateEncoder(VariableConcat):
    def __init__(self, config: Config,
                 restriction: Optional[Iterable[str]] = None):
        if restriction is None:
            names = config.env.names_s
        else:
            names = sorted(set(restriction))
        super().__init__(config, names)

        dim = config.dims.actor_critic_hidden
        self.mlp = nn.Sequential(
            nn.Linear(self.size, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, **self.torchargs),
        )

    def forward(self, raw: Batch):
        x = super().forward(raw)
        e: torch.Tensor = self.mlp(x)
        return e


class Critic(BaseNN):
    def __init__(self, config: Config):
        super().__init__(config)

        self.state_encoder = StateEncoder(config)

        dim = self.dims.actor_critic_hidden
        self.readout = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim, 1, **self.torchargs),
        )

    def value(self, raw: Batch):
        x = self.state_encoder.forward(raw)
        x: torch.Tensor = self.readout(x)
        x = x.view(x.shape[0])
        return x

    def forward(self, raw: Batch):
        return self.value(raw)
    
    def compute(self, buffer: Buffer):
        with torch.no_grad():
            transitions = buffer.transitions[:]
            terminated = transitions.terminated
            r = transitions.rewards

            # compute values
            v = self.value(transitions)
            v_next = self.value(self.shift_states(transitions))
            
            # compute TD-residuals
            gamma = self.config.rl_args.discount
            v_next = torch.masked_fill(v_next, terminated, 0.)
            td_residual = (r + gamma * v_next) - v

            # compute advantages using GAE
            done = transitions.done
            n = transitions.n
            w = self.config.rl_args.discount * self.config.rl_args.gae_lambda
            if w > 0:
                gae = torch.empty(n, dtype=DType.Numeric.torch,
                                  device=self.device)
                temp = 0
                for i in range(n-1, -1, -1):
                    if done[i]:
                        temp = td_residual[i]
                    else:
                        temp = w * temp + td_residual[i]
                    gae[i] = temp
            else:
                gae = td_residual
            
            # compute targets
            targets = v + gae

        if self.config.rl_args.use_adv_norm:
            std, mean = torch.std_mean(gae)
            adv = (gae - mean) / (std + 1e-8)
        else:
            adv = gae
        
        buffer[_ADV] = adv
        buffer[_TARGET] = targets
            

class Actor(BaseNN):
    def __init__(self, config: Config):
        super().__init__(config)

        self._akeys = self.env.names_a

        self.encoder = StateEncoder(config)
        self.decoders = {var: self.__make_decoder(var)
                         for var in self._akeys}

    def __make_decoder(self, var: str):
        decoder = DistributionDecoder(self.dims.actor_critic_hidden,
                                      self.v(var), self.config)
        self.add_module(f'{var} decoder', decoder)
        return decoder

    def forward(self, raw: Batch):
        e = self.encoder.forward(raw)
        out = Distributions(raw.n)
        for k, d in self.decoders.items():
            da = d.forward(e)
            out[k] = da
        return out


class PPO(Configured):
    def __init__(self, config: Config):
        super().__init__(config)
        self.args = self.config.rl_args

        self.actor = Actor(config)
        self.__old_actor = Actor(config)
        self.critic = Critic(config)
        self.actor.init_parameters()
        self.critic.init_parameters()
        self.opt_a = self.F.get_optmizer(self.args.optim_args, self.actor)
        self.opt_c = self.F.get_optmizer(self.args.optim_args, self.critic)
    
    def __old_actor_update(self):
        for name, param in self.actor.named_parameters():
            old_param = self.__old_actor.get_parameter(name)
            old_param.data[:] = param.data

    def actor_loss_entropy(self, data: Transitions):
        with torch.no_grad():
            adv = data[_ADV]
            old_policy = self.__old_actor.forward(data)
            actions = data.select(self.env.names_a).kapply(self.raw2label)
            old_logprob = old_policy.logprob(actions)

        b1 = self.args.kl_penalty
        b2 = self.args.entropy_penalty

        policy = self.actor.forward(data)
        logprob = policy.logprob(actions)
        importance = torch.exp(logprob - old_logprob)
        kl = old_policy.kl(policy)
        entropy = policy.entropy()

        j = importance * adv - b1 * kl + b2 * entropy
        assert u.TensorOperator.valid(j)
        return -torch.mean(j), float(torch.mean(entropy))

    def critic_loss(self, data: Transitions):
        with torch.no_grad():
            targets = data[_TARGET]
        
        value = self.critic.value(data)
        error = targets - value

        return torch.mean(torch.square(error))
    
    def act(self, states: NamedValues, mode=False):
        s =  Batch.from_sample(self.as_raws(states))
        pi = self.actor.forward(s)
        if mode:
            a = pi.mode()
        else:
            a = pi.sample()
        a = a.kapply(self.label2raw)
        a = self.as_numpy(a)
        return a
            
    def optimize(self, buffer: Buffer):
        batchsize = self.args.optim_args.batchsize
        loss_log = u.Log()

        # update old actor
        self.__old_actor_update()
        
        self.critic.compute(buffer)
        for i in range(self.args.n_epoch_critic):
            for data in buffer.epoch(batchsize):
                loss = self.critic_loss(data)
                loss_log['critic'] = float(loss)
                loss.backward()
                self.F.optim_step(self.args.optim_args, self.critic,
                                  self.opt_c)
        
        self.critic.compute(buffer)
        for i in range(self.args.n_epoch_actor):
            for data in buffer.epoch(batchsize):
                loss, entropy = self.actor_loss_entropy(data)
                loss_log['actor'] = float(loss) 
                loss_log['entropy'] = entropy
                loss.backward()
                self.F.optim_step(self.args.optim_args, self.actor,
                                  self.opt_a)

        del buffer[_ADV]
        del buffer[_TARGET]

        return loss_log
    
    def show_loss(self, log: u.Log):
        u.Log.figure(figsize=(12, 5))  # type: ignore
        u.Log.subplot(121, title='actor loss')
        log['actor'].plot(color='k')
        u.Log.subplot(122, title='critic loss')
        log['critic'].plot(color='k')
        u.Log.show()
