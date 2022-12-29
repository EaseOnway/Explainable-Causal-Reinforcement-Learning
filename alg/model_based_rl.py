from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import torch
import numpy as np
import random

from .train import Train

from core import Batch, Transitions, Tag
from learning.buffer import Buffer
from learning.planning import PPO, Actor
from learning.env_model import RolloutGenerator

from utils import Log, RewardScaling
import utils



_LL = 'loglikelihood'
_NLL_LOSS = 'NLL loss'
_ACTOR_LOSS = 'actor loss'
_CRITIC_LOSS = 'critic loss'
_ACTOR_ENTROPY = 'actor entropy'
_REWARD = 'reward'
_RETURN = 'return'


class ModelBasedRL(Train):
    use_existing_path = False

    @classmethod
    def init_parser(cls, parser):
        super().init_parser(parser)
        parser.add_argument('--ablation', type=str)
        parser.add_argument('--n-step', type=int, default=200)

    def make_title(self):
        title = "model-based"
        ablation = self.args.ablation
        if self.args.ablation is None:
            return title
        else:
            return title + '-' + ablation

    def configure(self):
        config = super().configure()

        # setting ablation
        ablation = self.args.ablation
        if ablation == 'recur':
            config.ablations.recur = True
        elif ablation == 'offline':
            config.ablations.offline = True
        elif ablation == 'dense':
            config.ablations.dense = True
        elif ablation == 'mlp':
            config.ablations.mlp = True
        elif ablation is not None:
            raise ValueError(f"unsupported ablation: {ablation}")
        
        return config
    
    def setup(self):
        super().setup()
        context = self.context
        # planning algorithm
        self.ppo = PPO(context)
        self.ppo.actor.init_parameters()
        self.ppo.critic.init_parameters()
        # buffer for causal model
        self.buffer_m = Buffer(context, self.config.model.buffer_size)
        # statistics for reward scaling
        self.reward_scaling = RewardScaling(self.config.rl.discount)
        # env model
        self.env_models = self.creat_env_models(self.config.mbrl.ensemble_size)
        self.env_models.init_parameters()
        self.env_model_optimizers = self.env_models.optimizers()

        # runtime variable
        self.causal_interval = self.config.mbrl.causal_interval_min
        self.graph_next_update = 0
        self.i_step = 0
        self.n_step: int = self.args.n_step
    
    def dream(self, buffer: Buffer, n_sample: int, len_rollout: int):
        '''generate samples into the buffer using the environment model'''
        log = Log()

        batchsize = self.config.mbrl.dream_batch_size
        env_m = RolloutGenerator(self.env_models, self.buffer_m, len_rollout)
        env_m.reset(batchsize)
        
        tr: List[Transitions] = []
        
        for i_sample in range(0, n_sample, batchsize):
            with torch.no_grad():
                s = env_m.current_state
                a = self.ppo.actor.forward(s).sample().kapply(self.label2raw)
                tran = env_m.step(a)
            tr.append(tran)
        
        i_sample = 0
        for i in range(batchsize):
            rj = range(0, min(len(tr), n_sample-i_sample))
            i_sample += rj.stop

            data = {name: torch.stack([tr[j][name][i] for j in rj])
                    for name in self.env.names_all}
            reward = torch.stack([tr[j].rewards[i]  for j in rj])
            code = torch.stack([tr[j].tagcode[i] for j in rj])
            code[-1] |= Tag.TRUNCATED.mask

            for r in reward:
                log[_REWARD] = float(r)

            if self.config.rl.use_reward_scaling:
                reward /= self.reward_scaling.std

            t = Transitions(data, reward, code)
            buffer.append(t)
            
        assert i_sample == n_sample
        return log

    def __step(self):
        i_step, n_step = self.i_step, self.n_step
        config = self.config
        print(f"---------------step {i_step} / {n_step}----------------")

        # planing buffer
        buffer_p = Buffer(self.context, config.mbrl.n_sample_rollout)

        # collecting explore samples
        n_sample_explore = config.mbrl.n_sample_explore
        self.collect(self.buffer_m, n_sample_explore, None,
                     self.config.rl.use_reward_scaling, self.ppo.actor)
        # collecting exploit samples
        n_sample_exploit = config.mbrl.n_sample_exploit
        log_step = self.collect(self.buffer_m, n_sample_exploit, 0,
                                self.config.rl.use_reward_scaling)
        self.n_sample += n_sample_exploit + n_sample_exploit
        true_reward = log_step[_REWARD].mean
        true_return = log_step[_RETURN].mean

        print(f"episodic return:\t{true_return}")
        print(f"mean reward:\t{true_reward} (truth)")

        # fit causal equation
        if not self.ablations.offline\
                and i_step >= self.graph_next_update:
            self.causal_discovery()
            self.graph_next_update = int(max(i_step + 1, i_step + self.causal_interval))
            self.causal_interval = min(
                self.causal_interval + self.config.mbrl.causal_interval_increase,
                self.config.mbrl.causal_interval_max)
            print(f"next causal discovery will be at step {self.graph_next_update}")

            _, fit_eval = self.fit(config.mbrl.n_batch_fit_new_graph * config.mbrl.ensemble_size)
        else:
            _, fit_eval = self.fit(config.mbrl.n_batch_fit * config.mbrl.ensemble_size)

        # compute rollout length
        if isinstance(config.mbrl.rollout_length, int):
            len_rollout = config.mbrl.rollout_length
        else:
            a, b = config.mbrl.rollout_length
            len_rollout = int(a + (b - a)*i_step/n_step)
        print(f"use rollout length: {len_rollout}")

        # planning
        plan_log = Log()
        for i_round in range(self.config.mbrl.n_round_planning):
            # dream samples
            buffer_p.clear()
            dream_log = self.dream(buffer_p, buffer_p.max_size, len_rollout)
            plan_log[_REWARD] = dream_log[_REWARD].mean

            # planning
            loss = self.ppo.optimize(buffer_p)
            plan_log[_ACTOR_LOSS] = actor_loss = loss['actor'].mean
            plan_log[_CRITIC_LOSS] = critic_loss = loss['critic'].mean
            plan_log[_ACTOR_ENTROPY] = actor_entropy = loss['entropy'].mean
            print("round %d: actor loss = %f, critic loss= %f, actor entropy = %f" %
                    (i_round, actor_loss, critic_loss, actor_entropy))

        dream_reward = plan_log[_REWARD].data[0]
        actor_loss = plan_log[_ACTOR_LOSS].data[-1]
        critic_loss = plan_log[_CRITIC_LOSS].data[-1]
        actor_entropy = plan_log[_ACTOR_ENTROPY].data[-1]
        
        # write summary
        writer = self.writer
        writer.add_scalar('reward', true_reward, self.n_sample)
        if not np.isnan(true_return):
            writer.add_scalar('return', true_return, self.n_sample)
        writer.add_scalar('reward_dreamed', dream_reward, self.n_sample)
        writer.add_scalar('actor_loss', actor_loss, self.n_sample)
        writer.add_scalar('critic_loss', critic_loss, self.n_sample)
        writer.add_scalar('actor_entropy', actor_entropy, self.n_sample)
        writer.add_scalar('fitting_loss', fit_eval[_NLL_LOSS].mean, self.n_sample)
        writer.add_scalars('log_likelihood',
                           {k: fit_eval[_LL, k].mean
                            for k in self.env.names_output},
                           self.n_sample)

        # show info
        print(f"actor loss:\t{actor_loss}")
        print(f"critic loss:\t{critic_loss}")
        print(f"episodic return:\t{true_return}")
        print(f"mean reward:\t{true_reward} (truth); {dream_reward} (dream)")
        print(f"actor entropy:\t{actor_entropy}")

        # save
        self.save_all()
    
    def save_all(self):
        # save
        self.save(self.ppo.actor.state_dict(), "actor", "nn")
        self.save(self.ppo.critic.state_dict(), "critic", "nn")
        for i, model in enumerate(self.env_models):
            self.save(model.state_dict(), f'env-model-{i}', 'nn')
        if not self.config.ablations.mlp:
            self.save(self.causal_graph, 'causal-graph', 'json')

    def main(self):
        print("warming up")
        self.warmup(self.config.mbrl.n_sample_warmup, 1.)
        for i_step in range(self.n_step):
            self.i_step = i_step
            self.__step()
