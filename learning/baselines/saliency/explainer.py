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


class SaliencyExplainner(Configured):

    def __init__(self, trainer: Train):
        super().__init__(trainer.config)
        self.args = self.config.saliency

        self.trainer = trainer
        self.expert_actor = trainer.ppo.actor
        self.masked_actor = SaliencyActor(self.config)
        self.buffer = Buffer(self.config, self.args.buffer_size)

        self.opt = self.F.get_optmizer(self.args.optim_args, self.masked_actor)
    
    def __batch(self, batch: Transitions):
        state = batch.select(self.env.names_s)
        expert_actions = batch.select(self.env.names_a)
        masked_pi = self.masked_actor.forward(state)
        mask = self.masked_actor.mask
        log_prob = masked_pi.logprob(expert_actions)

        nll_loss = -torch.mean(log_prob)
        sparity_loss = torch.mean(torch.sum(torch.abs(mask), dim=1))
        loss = nll_loss + self.args.sparse_factor * sparity_loss

        loss.backward()
        self.F.optim_step(self.args.optim_args, self.masked_actor, self.opt)
        
        return float(nll_loss), float(sparity_loss)

    def __epoch(self, n_batch: int):
        if self.trainer.show_detail:
            print("collecting samples...")
        self.trainer.collect(self.buffer, self.buffer.max_size, 0.)

        log = Log()
        for i in range(n_batch):
            batch = self.buffer.sample_batch(self.args.optim_args.batchsize)
            nll_loss, sparity_loss = self.__batch(batch)
            log['nll_loss'] = nll_loss
            log['sparity_loss'] = sparity_loss
        
            if self.trainer.show_detail:
                interval = n_batch // 10
                if interval == 0 or (i + 1) % interval == 0:
                    print(f"  batch {i + 1}/{n_batch}:\n"
                          f"    nll_loss = {nll_loss}\n"
                          f"    sparity_loss = {sparity_loss}")
        
        return log

    def load(self, path: Optional[str] = None):
        path = path or self.trainer.run_dir + "saliency_model"
        self.masked_actor.load_state_dict(torch.load(path))

    def save(self, path: Optional[str] = None):
        path = path or self.trainer.run_dir + "saliency_model"
        torch.save(self.masked_actor.state_dict(), path)

    def train(self, n_epoch):
        for i in range(n_epoch):
            print(f'epoch {i + 1} begins')
            log = self.__epoch(self.args.n_batch_per_epoch)

            if self.trainer.show_detail:
                print(f"  epoch {i + 1} completed:\n"
                          f"    nll_loss = {log['nll_loss'].mean}\n"
                          f"    sparity_loss = {log['sparity_loss'].mean}")

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

        print("I take the action with log-probability %.4f:" % logprob)
        for name, x in action.items():
            print(f"|\t{name} = {self.v(name).text(x)}")
        print("because the saliency of state is")
        for name, x in state.items():
            print(f"|\t%.4f: {name} = {self.v(name).text(x)}" % mask[name])
