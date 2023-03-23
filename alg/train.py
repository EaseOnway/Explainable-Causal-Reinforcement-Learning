import os
from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import numpy as np
import random
import torch

import json

import tensorboardX

from core import Batch, Transitions, Tag
from learning.buffer import Buffer
from learning.causal_discovery import discover
from learning.planning import PPO, Actor
from learning.env_model import RolloutGenerator, CausalEnvModel, MLPEnvModel,\
    EnvModelEnsemble, EnvModel

from utils import Log, RewardScaling
from utils.typings import ParentDict, NamedArrays, ParentDict
import utils

from .experiment import Experiment

_LL = 'loglikelihood'
_NLL_LOSS = 'NLL loss'
_ACTOR_LOSS = 'actor loss'
_CRITIC_LOSS = 'critic loss'
_ACTOR_ENTROPY = 'actor entropy'
_REWARD = 'reward'
_RETURN = 'return'


class Train(Experiment):

    def setup(self):
        log_path = self._file_path("log")
        if not log_path.exists():
            os.makedirs(log_path)
        
        self.writer = tensorboardX.SummaryWriter(str(log_path))

        # probable attributes
        # model
        self.env_models: EnvModelEnsemble
        self.env_model_optimizers: Tuple[torch.optim.Optimizer, ...]
        # planning algorithm
        self.ppo: PPO
        # buffer for environment model
        self.buffer_m: Buffer
        # statistics for reward scaling
        self.reward_scaling: RewardScaling
        
        # declear runtime variables
        self.n_sample = 0
    
    def creat_env_models(self, ensemble_size: int) -> EnvModelEnsemble:
        if self.config.ablations.mlp:
            return EnvModelEnsemble(self.context, 
                tuple(MLPEnvModel(self.context) for _ in range(ensemble_size)))
        else:
            return EnvModelEnsemble(self.context, 
                tuple(CausalEnvModel(self.context) for _ in range(ensemble_size)))
    
    def collect(self, buffer: Buffer, n_sample: int, explore_rate: Optional[float],
                reward_scaling: bool, actor: Optional[Actor] = None):
        '''collect real-world samples into the buffer, and compute returns'''

        def get_explore_rate():
            return random.uniform(0, self.config.mbrl.explore_rate_max)

        log = Log()
        max_len = self.config.rl.max_episode_length
        self.env.reset()
        episodic_return = 0.
        actor = actor or self.ppo.actor

        if explore_rate is None:
            explore_rate = get_explore_rate()
            _explore_rate_fixed = False
        else:
            _explore_rate_fixed = True

        for i_sample in range(n_sample):
            initiated = self.env.t == 0

            # interact with the environment
            if utils.Random.event(explore_rate):
                a = self.env.random_action()
            else:
                a = actor.act(self.env.current_state, False)

            transition = self.env.step(a)
            reward = transition.reward
            terminated = transition.terminated
            truncated = (self.env.t == max_len)

            # reset environment
            if truncated:
                self.env.reset()

            # record information
            episodic_return += reward
            log[_REWARD] = reward
            if truncated or terminated:
                log[_RETURN] = episodic_return
                episodic_return = 0.
            
            if (truncated or terminated) and not _explore_rate_fixed:
                explore_rate = get_explore_rate()

            # truncate the last sample
            if i_sample == n_sample - 1:
                truncated = True

            # write buffer
            tagcode = Tag.encode(terminated, truncated, initiated)
            if reward_scaling:
                reward = self.reward_scaling(reward, (truncated or terminated))
            buffer.write(transition.variables, reward, tagcode)

        return log

    @property
    @final
    def causal_graph(self) -> ParentDict:
        return self.__causal_graph

    @causal_graph.setter
    @final
    def causal_graph(self, graph: ParentDict):
        self.__causal_graph = graph
        if not self.config.ablations.mlp:
            assert(isinstance(self.env_models, CausalEnvModel) or
                   isinstance(self.env_models, EnvModelEnsemble))
            self.env_models.load_graph(self.__causal_graph)
    
    def evaluate_policy(self, n_sample: int):
        '''collect real-world samples into the buffer, and compute returns'''

        log = Log()
        max_len = self.config.rl.max_episode_length
        self.env.reset()
        episodic_return = 0.

        for i_sample in range(n_sample):
            a = self.ppo.actor.act(self.env.current_state, False)
            transition = self.env.step(a)
            reward = transition.reward
            terminated = transition.terminated
            truncated = (self.env.t == max_len)

            # reset environment
            if truncated:
                self.env.reset()

            # record information
            episodic_return += reward
            log[_REWARD] = reward
            if truncated or terminated:
                log[_RETURN] = episodic_return
                episodic_return = 0.

        return log

    def plot_causal_graph(self, format='png'):
        return utils.visualize.plot_digraph(
            self.env.names_input + self.env.names_output,
            self.__causal_graph, format=format)  # type: ignore
    
    def save(self, o: object, name: str, fmt: Optional[str] = None):
        path = self._file_path(name, fmt)
        if fmt == 'json':
            with path.open('w') as f:
                json.dump(o, f, indent=4)
        else:
            torch.save(o, path)

        print(f"successfully saved: {path}")

    def load(self, name: str, fmt: Optional[str] = None):
        path = self._file_path(name, fmt)
        if fmt == 'json':
            with path.open('r') as f:
                o = json.load(f)
        else:
            o = torch.load(path)

        print(f"successfully loaded: {path}")
        return o
    
    def _get_data_for_causal_discovery(self) -> NamedArrays:
       temp = self.buffer_m.tensors[:]
       temp = {k: self.raw2input(k, v).numpy() for k, v in temp.items()}
       return temp

    def warmup(self, n_sample: int, explore_rate: Optional[float]):
        return self.collect(self.buffer_m, n_sample, explore_rate,
                            self.config.rl.use_reward_scaling)
    
    def update_variable_normalizer(self):
        for name in self.env.names_input:
            data = self.buffer_m[name].to(device=self.device)
            value = self.v(name).raw2input(data)
            std, mean = torch.std_mean(value, dim=0)
            for model in self.env_models:
                if isinstance(model, CausalEnvModel):
                    model.encoder[name].load_std_mean(std, mean)

    def __fit_batch(self, transitions: Transitions, eval=False):
        i, net = self.env_models.random_select()
        opt = self.env_model_optimizers[i]
        lls = net.get_loglikeli_dic(transitions)
        ll = net.loglikelihood(lls)
        loss = -ll
        if not eval:
            loss.backward()
            self.F.optim_step(self.config.model.optim, net, opt)
        return float(loss), {k: float(e) for k, e in lls.items()}

    def fit_epoch(self, buffer: Buffer, log: Log, eval=False):
        '''
        train network with fixed causal graph.
        '''
        args = self.config.model.optim
        for batch in buffer.epoch(args.batchsize):
            loss, lls = self.__fit_batch(batch, eval)
            log[_NLL_LOSS] = loss
            for k, ll in lls.items():
                log[_LL, k] = ll
    
    def fit(self, n_batch: int, converge_interval=-1):
        # fit causal equation
        train_log = Log()
        batch_size = self.config.model.optim.batchsize
        interval = n_batch // 20

        print(f"setting up normalizer")
        self.update_variable_normalizer()

        print(f"start fitting...")
        self.env_models.train(True)
        for i_batch in range(n_batch):
            batch = self.buffer_m.sample_batch(batch_size)
            nll, loglikelihoods = self.__fit_batch(batch, eval=False)
            train_log[_NLL_LOSS] = nll
            for k, ll in loglikelihoods.items():
                train_log[_LL, k] = ll

            if interval == 0 or (i_batch + 1) % interval == 0:
                print(f"batch {i_batch + 1}/{n_batch}: loss = {nll}")
            
            # check converge
            if converge_interval > 0 and i_batch >= converge_interval\
                    and (i_batch + 1) % converge_interval == 0:
                data = train_log[_NLL_LOSS].data
                window1 = data[-2*converge_interval: -converge_interval]
                window2 = data[-converge_interval:]
                if np.mean(window2) >= np.mean(window1):
                    print("converge detected. stop iteration.")
                    break

        # evaluate
        self.env_models.train(False)
        eval_log = Log()
        self.fit_epoch(self.buffer_m, eval_log, eval=True)
        
        # show info
        print(f"nll-loss:\t{eval_log[_NLL_LOSS].mean}")
        for k in self.env.names_output:
            print(f"log-likelihood of '{k}':\t{eval_log[_LL, k].mean}")
        
        return train_log, eval_log

    def causal_discovery(self):
        # causal discovery
        if self.config.ablations.dense:
            self.causal_graph = self.env.get_full_graph()
        elif self.config.ablations.mlp:
            pass
        else:
            data = self._get_data_for_causal_discovery()

            # b = self.config.model.pthres_max
            # a = self.config.model.pthres_min
            # assert a <= b
            # n_orc = self.config.model.n_sample_oracle
            # n = len(self.buffer_m)
            # pthres = float(np.clip(b - (n / n_orc) * (b - a), a, b))
            pthres = self.config.model.pthres
            print(f"perform causal disocery with threshold {pthres}")
        
            self.causal_graph = discover(
                data, self.env, pthres,
                True, self.config.model.n_jobs_fcit)
