from typing import Optional, Sequence, List

import numpy as np
import torch

from learning.train import Train
from learning.buffer import Buffer
from core.data import Transitions, Batch
from utils.typings import NamedValues
from utils import Log
from learning.base import Configured

from .actor import SaliencyActor
from .q_net import QNet


class BaselineExplainner(Configured):

    def __init__(self, trainer: Train):
        super().__init__(trainer.config)

        self.trainer = trainer
        self.expert_actor = trainer.ppo.actor
        self.masked_actor = SaliencyActor(self.config)
        self.qnet = QNet(self.config)

        self.args = self.config.baseline
        self.opt_saliency = self.F.get_optmizer(
            self.args.optim_args, self.masked_actor)
        self.opt_q = self.F.get_optmizer(
            self.args.optim_args, self.qnet)

    def __batch_q(self, batch: Transitions):
        q = self.qnet(batch)
        r = batch.rewards
        gamma = self.config.rl_args.discount
        with torch.no_grad():
            next_batch = self.shift_states(batch)
            next_pi = self.expert_actor.forward(next_batch)
            next_a = next_pi.sample()
            next_batch.update(next_a)
            next_q = self.qnet.forward(next_batch)
            target = torch.where(batch.terminated, r, r + gamma * next_q)
        td_error = torch.mean(torch.square(target - q))
        td_error.backward()
        self.F.optim_step(self.args.optim_args, self.qnet, self.opt_q)
        return float(td_error)

    def __batch_saliency(self, batch: Transitions):
        state = batch.select(self.env.names_s)
        expert_actions = batch.select(self.env.names_a)
        masked_pi = self.masked_actor.forward(state)
        mask = self.masked_actor.mask
        log_prob = masked_pi.logprob(expert_actions)

        nll_loss = -torch.mean(log_prob)
        sparity_loss = torch.mean(torch.sum(torch.abs(mask), dim=1)) / self.env.num_s
        loss = nll_loss + self.args.sparse_factor * sparity_loss

        loss.backward()
        self.F.optim_step(self.args.optim_args, self.masked_actor, self.opt_saliency)
        
        return float(nll_loss), float(sparity_loss)
    
    def train_saliency(self, n_sample: int, n_batch: int):
        log = Log()
        buffer = Buffer(self.config, n_sample)

        print("collecting sample")
        self.trainer.collect(buffer, n_sample, 0., False)

        for i in range(n_batch):
            batch = buffer.sample_batch(self.args.optim_args.batchsize)
            nll_loss, sparity_loss = self.__batch_saliency(batch)
            log['nll_loss'] = nll_loss
            log['sparity_loss'] = sparity_loss
        
            if self.trainer.show_detail:
                interval = n_batch // 10
                if interval == 0 or (i + 1) % interval == 0:
                    print(f"  batch {i + 1}/{n_batch}:\n"
                          f"    nll_loss = {nll_loss}\n"
                          f"    sparity_loss = {sparity_loss}\n")

        print(f"completed:\n"
              f"    nll_loss = {log['nll_loss'].mean}\n"
              f"    sparity_loss = {log['sparity_loss'].mean}")
        
        return log

    def train_q(self, n_sample: int, n_batch: int):
        log = Log()
        buffer = Buffer(self.config, n_sample)

        print("collecting sample")
        self.trainer.collect(buffer, n_sample, None, False)

        for i in range(n_batch):
            batch = buffer.sample_batch(self.args.optim_args.batchsize)
            td_error = self.__batch_q(batch)
            log['td_error'] = td_error
        
            if self.trainer.show_detail:
                interval = n_batch // 10
                if interval == 0 or (i + 1) % interval == 0:
                    print(f"  batch {i + 1}/{n_batch}:\n"
                          f"    td_error = {td_error}")

        print(f"completed:\n"
              f"    td_error = {log['td_error'].mean}")
        
        return log

    def state_dict(self):
        return {"saliency": self.masked_actor.state_dict(),
                "q": self.qnet.state_dict()}
    
    def load_state_dict(self, dic: dict):
        self.masked_actor.load_state_dict(dic['saliency'])
        self.qnet.load_state_dict(dic['q'])
    
    def save(self, path: Optional[str] = None):
        path = path or self.trainer.run_dir + 'saved-baselines'
        torch.save(self.state_dict(), path)
        print(f"succesfully saved [baselines] to {path}")
    
    def load(self, path: Optional[str] = None):
        path = path or self.trainer.run_dir + 'saved-baselines'
        self.load_state_dict(torch.load(path))
        print(f"succesfully load [baselines] from {path}")
                
    def __compute_importance(self, state: NamedValues):
        n = self.config.baseline.n_sample_importance

        s = Batch.from_sample(self.named_tensors(state))

        with torch.no_grad():
            a_pi = self.expert_actor.forward(s).mode().kapply(self.label2raw)

            a_rd = [self.env.random_action() for _ in range(n)]
            a = {name: torch.tensor([x[name] for x in a_rd],
                                    device=self.device,
                                    dtype=self.v(name).dtype.torch)
                    for name in self.env.names_a}
            a = {name: torch.cat((a_pi[name], a[name]), dim=0)
                for name in self.env.names_a}
            s = {name: torch.broadcast_to(s[name], (n+1, *self.v(name).shape))
                for name in self.env.names_s}
            b = Batch(n+1, s)
            b.update(a)

            q = self.qnet.forward(b)
        
        return float(torch.max(q) - torch.min(q))

    def why(self, state: NamedValues, action: NamedValues):
        s = Batch.from_sample(self.named_tensors(state))
        a = Batch.from_sample(self.named_tensors(action))

        with torch.no_grad():
            pi = self.masked_actor.forward(s)
            logprob = float(pi.logprob(a))
            mask = self.T.t2a(self.masked_actor.mask.squeeze(0), float)   
            mask = {
                name: mask[i] for i, name in enumerate(self.env.names_s)
            }
            importance = self.__compute_importance(state)
            q = float(self.qnet.q(state, action))
            entropy = float(pi.entropy())

        print("I take the action with probability %.4f:" % np.exp(logprob))
        for name, x in action.items():
            print(f"|\t{name} = {self.v(name).text(x)}")
        print(f"which produces a Q-value of {q}")
        print("I make this decision since I notice that")
        for name, x in state.items():
            print(f"|\t%.4f: {name} = {self.v(name).text(x)}" % mask[name])
        print(f"the importance of this decision is {importance}")
        print(f"the uncertainty (entropy) of my policy is {entropy}")
