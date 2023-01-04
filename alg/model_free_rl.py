from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import torch
import numpy as np

from .train import Train

from core import Batch, Transitions, Tag
from learning.buffer import Buffer
from learning.planning import PPO
from learning.env_model import RolloutGenerator

from utils import Log, RewardScaling


_REWARD = 'reward'
_RETURN = 'return'


class ModelFreeRL(Train):
    use_existing_path = False

    @classmethod
    def init_parser(cls, parser):
        super().init_parser(parser)
        parser.add_argument('--n-step', type=int, default=200)
        parser.add_argument('--store-buffer', action='store_true', default=False)
    
    def make_title(self):
        return "model-free"
    
    def configure(self):
        config = super().configure()
        if self.args.config is None:
            config.rl.n_epoch_actor = 8
            config.rl.n_epoch_critic = 64
        return config

    def setup(self):
        super().setup()

        self.store_buffer: bool = self.args.store_buffer
        if self.store_buffer:
            self.buffer_m = Buffer(self.context, self.config.model.buffer_size)

        # planning algorithm
        self.ppo = PPO(self.context)
        self.ppo.actor.init_parameters()
        self.ppo.critic.init_parameters()
        # statistics for reward scaling
        self.reward_scaling = RewardScaling(self.config.rl.discount)
    
    def __step(self, i_step: int, n_step: int):
        print(f"---------------step {i_step} / {n_step}----------------")

        # planing buffer
        buffer_p = Buffer(self.context, self.config.rl.n_sample)
        
        # collect samples
        buffer_p.clear()
        log = self.collect(buffer_p, buffer_p.max_size, 0.,
                           self.config.rl.use_reward_scaling)
        self.n_sample += buffer_p.max_size
        true_reward = log[_REWARD].mean
        true_return = log[_RETURN].mean

        if self.store_buffer:
            self.buffer_m.append(
                buffer_p.sample_batch(self.config.mbrl.n_sample_exploit))

        # planning
        plan_loss = self.ppo.optimize(buffer_p)
        actor_loss = plan_loss['actor'].mean
        critic_loss = plan_loss['critic'].mean
        actor_entropy = plan_loss['entropy'].mean

        # write summary
        writer = self.writer
        writer.add_scalar('reward', true_reward, self.n_sample)
        if not np.isnan(true_return):
            writer.add_scalar('return', true_return, self.n_sample)
        writer.add_scalar('actor_loss', actor_loss, self.n_sample)
        writer.add_scalar('critic_loss', critic_loss, self.n_sample)
        writer.add_scalar('actor_entropy', actor_entropy, self.n_sample)

        # show info
        print(f"actor loss:\t{plan_loss['actor'].mean}")
        print(f"critic loss:\t{plan_loss['critic'].mean}")
        print(f"episodic return:\t{true_return}")
        print(f"mean reward:\t{true_reward}")
        print(f"actor entropy:\t{actor_entropy}")

        # save
        self.save_all()
    
    def save_all(self):
        # save
        self.save(self.ppo.actor.state_dict(), "actor", "nn")
        self.save(self.ppo.critic.state_dict(), "critic", "nn")
        if self.store_buffer:
            self.save(self.buffer_m.state_dict(), "data-buffer")

    def main(self):
        n_step: int = self.args.n_step
        for i_step in range(n_step):
            self.__step(i_step, n_step)
