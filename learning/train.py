import os
from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import numpy as np
import random
import torch
import cmd

import tensorboardX
from tensorboard.backend.event_processing import event_accumulator

from core import Batch, Transitions, Tag
from .buffer import Buffer
from .causal_discovery import discover, update
from .env_model import SimulatedEnvParallel, CausalNet, MLPNet,\
    EnvNetEnsemble, EnvModelNet
from .config import Config
from .base import Configured
from .planning import PPO
from utils import Log, RewardScaling
from utils.typings import ParentDict, NamedArrays
import utils


_LL = 'loglikelihood'
_NLL_LOSS = 'NLL loss'
_ACTOR_LOSS = 'actor loss'
_CRITIC_LOSS = 'critic loss'
_ACTOR_ENTROPY = 'actor entropy'
_REWARD = 'reward'
_EXP_ROOT = './experiments/'
_CONFIG_TXT = 'config.txt'
_RETURN = 'return'


class Train(Configured):

    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __init__(self, config: Config, name: str,
                 showinfo: Literal[None, 'brief', 'verbose', 'plot'] = 'verbose'):
        super().__init__(config)
        print("Using following configuration:")
        print(self.config)

        # arguments
        self.causal_args = self.config.envmodel_args
        self.rl_args = self.config.rl_args
        self.name = name
        self.dir = _EXP_ROOT + str(self.env) + '/' + self.name + '/'
        self.show_loss = (showinfo is not None)
        self.show_detail = (showinfo == 'verbose' or showinfo == 'plot')
        self.show_plot = (showinfo == 'plot')

        # causal graph, env model, and model optimizer
        def get_env_net() -> EnvModelNet:
            if self.config.ablations.mlp:
                return MLPNet(config)
            else:
                return CausalNet(config)
        self.__causal_graph: ParentDict
        if self.causal_args.n_ensemble <= 1:
            self.envnet = get_env_net()
        else:
            _nets = (get_env_net() for _ in range(self.causal_args.n_ensemble))
            self.envnet = EnvNetEnsemble(config, tuple(_nets))
        self.causopt = self.F.get_optmizer(self.causal_args.optim_args, self.envnet)

        # planning algorithm
        self.ppo = PPO(config)

        # buffer for causal model
        self.buffer_m = Buffer(config, self.causal_args.buffer_size)

        # statistics for reward scaling
        self.__reward_scaling = RewardScaling(self.rl_args.discount)
        
        # declear runtime variables
        self.__n_sample: int
        self.__episodic_return: float
        self.__run_dir: str
        self.__writer: tensorboardX.SummaryWriter
    
    @property
    def run_dir(self):
        return self.__run_dir
    
    @property
    @final
    def causal_graph(self):
        return self.__causal_graph

    @causal_graph.setter
    @final
    def causal_graph(self, graph: ParentDict):
        self.__causal_graph = graph
        if not self.config.ablations.mlp:
            assert(isinstance(self.envnet, CausalNet) or
                   isinstance(self.envnet, EnvNetEnsemble))
            self.envnet.load_graph(self.__causal_graph)

    def collect(self, buffer: Buffer, n_sample: int,
                explore_rate: Optional[float], eval=False):
        '''collect real-world samples into the buffer, and compute returns'''

        def get_explore_rate():
            return random.uniform(0, self.causal_args.explore_rate_max)

        log = Log()
        max_len = self.causal_args.maxlen_truth
        self.env.reset()
        self.__episodic_return = 0.

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
                a = self.ppo.act(self.env.current_state, False)

            transition = self.env.step(a)
            reward = transition.reward
            terminated = transition.terminated
            truncated = (self.env.t == max_len)

            # reset environment
            if truncated:
                self.env.reset()

            # record information
            self.__episodic_return += reward
            log[_REWARD] = reward
            if truncated or terminated:
                log[_RETURN] = self.__episodic_return
                self.__episodic_return = 0.
            
            if (truncated or terminated) and not _explore_rate_fixed:
                explore_rate = get_explore_rate()

            # truncate the last sample
            if i_sample == n_sample - 1:
                truncated = True
            
            # write buffer
            tagcode = Tag.encode(terminated, truncated, initiated)
            if self.rl_args.use_reward_scaling:
                reward = self.__reward_scaling(reward, (truncated or terminated))
            buffer.write(transition.variables, reward, tagcode)

        if not eval:
            self.__n_sample += n_sample

        return log

    def __dream(self, buffer: Buffer, n_sample: int):
        '''generate samples into the buffer using the environment model'''
        log = Log()

        max_len = self.causal_args.maxlen_dream
        batchsize = self.causal_args.dream_batch_size
        env_m = SimulatedEnvParallel(self.envnet, self.buffer_m, max_len)
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

            if self.rl_args.use_reward_scaling:
                reward /= self.__reward_scaling.std

            t = Transitions(data, reward, code)
            buffer.append(t)
            
        assert i_sample == n_sample
        return log
    
    def evaluate_policy(self, n_sample: int):
        '''collect real-world samples into the buffer, and compute returns'''

        log = Log()
        max_len = self.causal_args.maxlen_truth
        self.env.reset()
        self.__episodic_return = 0.

        for i_sample in range(n_sample):
            a = self.ppo.act(self.env.current_state, False)
            transition = self.env.step(a)
            reward = transition.reward
            terminated = transition.terminated
            truncated = (self.env.t == max_len)

            # reset environment
            if truncated:
                self.env.reset()

            # record information
            self.__episodic_return += reward
            log[_REWARD] = reward
            if truncated or terminated:
                log[_RETURN] = self.__episodic_return
                self.__episodic_return = 0.

        return log

    def __init_params(self):
        self.ppo.actor.init_parameters()
        self.ppo.critic.init_parameters()
        self.envnet.init_parameters()
        self.causal_graph = {j: {i for i in self.env.names_inputs
                                 if utils.Random.event(self.causal_args.prior)}
                             for j in self.env.names_outputs}

    def plot_causal_graph(self, format='png'):
        return utils.visualize.plot_digraph(
            self.env.names_inputs + self.env.names_outputs,
            self.__causal_graph, format=format)  # type: ignore
    
    __SAVE_KEY = Literal['agent', 'env-model', 'causal-graph',
                         'runtime', 'buffer']

    def save_item(self, key: __SAVE_KEY, path: Optional[str] = None):
        path = path or (self.__run_dir + 'saved-' + key)
        if key == 'agent':
            torch.save({'actor': self.ppo.actor.state_dict(),
                        'critic': self.ppo.critic.state_dict()}, path)
        elif key == 'env-model':
            torch.save({'inference-network': self.envnet.state_dict(),
                        'causal-graph': self.__causal_graph}, path)
        elif key == 'causal-graph':
            torch.save(self.__causal_graph, path)
        elif key == 'runtime':
            torch.save({'reward_scaling': self.__reward_scaling.state_dict(),
                        'n_sample': self.__n_sample}, path)
        elif key == 'buffer':
            self.buffer_m.save(path)
        else:
            raise NotImplementedError
        
        print(f"successfully saved [{key}] to {path}")

    def load_item(self, key: __SAVE_KEY, path: Optional[str] = None):
        path = path or (self.__run_dir + 'saved-' + key)
        saved = torch.load(path)
        if key == 'agent':
            self.ppo.actor.load_state_dict(saved['actor'])
            self.ppo.critic.load_state_dict(saved['critic'])
        elif key == 'env-model':
            self.envnet.load_state_dict(saved['inference-network'])
            self.causal_graph = saved['causal-graph']
        elif key == 'causal-graph':
            self.causal_graph = saved
        elif key == 'runtime':
            self.__reward_scaling.load_state_dict(saved['reward_scaling'])
            self.__n_sample = saved['n_sample']
        elif key == 'buffer':
            self.buffer_m.load(path)
        else:
            raise NotImplementedError
        
        print(f"successfully loaded [{key}] from {path}")
    
    def save_items(self, *keys: __SAVE_KEY):
        for key in set(keys):
            self.save_item(key)
    
    def load_items(self, *keys: __SAVE_KEY, skip_if_not_exist = False):
        if len(keys) == 0:
            keys = ('agent', 'env-model', 'causal-graph', 'runtime', 'buffer')
        for key in set(keys):
            try:
                self.load_item(key)
            except FileNotFoundError as e:
                if skip_if_not_exist:
                    pass
                else:
                    raise e
    
    def __get_data_for_causal_discovery(self) -> NamedArrays:
       temp = self.buffer_m.tensors[:]
       temp = {k: self.raw2input(k, v).numpy() for k, v in temp.items()}
       return temp
    
    def __get_run_dir(self, path: Optional[str] = None):
        if path is None:
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)
            run_names = os.listdir(self.dir)
            i = 0
            while True:
                i += 1
                run_name = "run-%d" % i
                if run_name not in run_names:
                    break
            path = self.dir + run_name + '/'
            os.makedirs(path)
        else:
            if len(path) == 0:
                path = './'
            elif path[-1] != '/' or path[-1] != '\\':
                path += '/'
            if not os.path.exists(path):
                os.makedirs(path)
        return path
    
    def init_run(self, dir: Optional[str] = None, resume=False,
                 n_sample: Optional[int] = None):
        path = self.__run_dir = self.__get_run_dir(dir)
        self.config.to_txt(path + _CONFIG_TXT)
        self.__n_sample = 0
        if resume:
            self.load_items(skip_if_not_exist=True)
            self.__writer = tensorboardX.SummaryWriter(
                path, purge_step=self.__n_sample)
        else:
            self.__init_params()
            self.__writer = tensorboardX.SummaryWriter(path)

        self.__episodic_return = 0.
        
        self.env.reset()
        self.buffer_m.clear()

    def warmup(self, n_sample: int, explore_rate: Optional[float]):
        return self.collect(self.buffer_m, n_sample, explore_rate)

    def __fit_batch(self, transitions: Transitions, eval=False):
        lls = self.envnet.get_loglikeli_dic(transitions)
        ll = self.envnet.loglikelihood(lls)
        loss = -ll
        if not eval:
            loss.backward()
            self.F.optim_step(self.causal_args.optim_args,
                              self.envnet, self.causopt)
        return float(loss), {k: float(e) for k, e in lls.items()}

    def __fit_epoch(self, buffer: Buffer, log: Log, eval=False):
        '''
        train network with fixed causal graph.
        '''
        args = self.causal_args.optim_args
        for batch in buffer.epoch(args.batchsize):
            loss, lls = self.__fit_batch(batch, eval)
            log[_NLL_LOSS] = loss
            for k, ll in lls.items():
                log[_LL, k] = ll

    def __show_fit_log(self, log: Log):
        Log.figure(figsize=(12, 5))  # type: ignore
        Log.subplot(121, title=_NLL_LOSS)
        log[_NLL_LOSS].plot(color='k')
        Log.subplot(122, title=_LL)
        log[_LL].plots(self.env.names_outputs)
        Log.show()
    
    def fit(self, n_batch: int, converge_interval=-1):
        # fit causal equation
        train_log = Log()
        batch_size = self.causal_args.optim_args.batchsize

        print(f"start fitting...")
        self.envnet.train(True)
        for i_batch in range(n_batch):
            batch = self.buffer_m.sample_batch(batch_size)
            nll, loglikelihoods = self.__fit_batch(batch, eval=False)
            train_log[_NLL_LOSS] = nll
            for k, ll in loglikelihoods.items():
                train_log[_LL, k] = ll

            if self.show_detail:
                interval = n_batch // 20
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
        self.envnet.train(False)
        eval_log = Log()
        self.__fit_epoch(self.buffer_m, eval_log, eval=True)
        
        # show info
        if self.show_loss:
            print(f"nll-loss:\t{eval_log[_NLL_LOSS].mean}")
        if self.show_detail:
            for k in self.env.names_outputs:
                print(f"log-likelihood of '{k}':\t{eval_log[_LL, k].mean}")
        if self.show_plot:
            self.__show_fit_log(train_log)
        
        return train_log, eval_log

    def causal_discovery(self):
        # causal discovery
        if self.config.ablations.dense:
            self.causal_graph = self.env.get_full_graph()
        elif self.config.ablations.mlp:
            pass
        else:
            data = self.__get_data_for_causal_discovery()
            self.causal_graph = discover(data, self.env,
                                        self.causal_args.pthres_independent,
                                        self.show_detail,
                                        self.causal_args.n_jobs_fcit)

    def __step_policy_model_based(self, i_step: int):
        # planing buffer
        buffer_p = Buffer(self.config, self.config.rl_args.buffer_size)

        # collect samples
        self.collect(self.buffer_m, self.causal_args.n_sample_collect, None)

        # evaluate policy
        log_step = self.evaluate_policy(self.causal_args.n_sample_evaluate_policy)
        true_reward = log_step[_REWARD].mean
        true_return = log_step[_RETURN].mean

        # fit causal equation
        if not self.ablations.offline\
                and i_step % self.causal_args.interval_graph_update == 0:
            self.causal_discovery()
            _, fit_eval = self.fit(self.causal_args.n_batch_fit_new_graph)
        else:
            _, fit_eval = self.fit(self.causal_args.n_batch_fit)

        # planning
        plan_log = Log()
        for i_round in range(self.rl_args.n_round_model_based):
            # dream samples
            buffer_p.clear()
            dream_log = self.__dream(buffer_p, buffer_p.max_size)
            plan_log[_REWARD] = dream_log[_REWARD].mean

            # planning
            loss = self.ppo.optimize(buffer_p)
            plan_log[_ACTOR_LOSS] = actor_loss = loss['actor'].mean
            plan_log[_CRITIC_LOSS] = critic_loss = loss['critic'].mean
            plan_log[_ACTOR_ENTROPY] = actor_entropy = loss['entropy'].mean
            if self.show_loss:
                print("round %d: actor loss = %f, critic loss= %f, actor entropy = %f" %
                        (i_round, actor_loss, critic_loss, actor_entropy))

        dream_reward = plan_log[_REWARD].data[0]
        actor_loss = plan_log[_ACTOR_LOSS].data[-1]
        critic_loss = plan_log[_CRITIC_LOSS].data[-1]
        actor_entropy = plan_log[_ACTOR_ENTROPY].data[-1]
        
        # write summary
        writer = self.__writer
        writer.add_scalar('reward', true_reward, self.__n_sample)
        if not np.isnan(true_return):
            writer.add_scalar('return', true_return, self.__n_sample)
        writer.add_scalar('reward_dreamed', dream_reward, self.__n_sample)
        writer.add_scalar('actor_loss', actor_loss, self.__n_sample)
        writer.add_scalar('critic_loss', critic_loss, self.__n_sample)
        writer.add_scalar('actor_entropy', actor_entropy, self.__n_sample)
        writer.add_scalar('fitting_loss', fit_eval[_NLL_LOSS].mean, self.__n_sample)
        writer.add_scalars('log_likelihood',
                           {k: fit_eval[_LL, k].mean
                            for k in self.env.names_outputs},
                           self.__n_sample)

        # show info
        if self.show_loss:
            print(f"actor loss:\t{actor_loss}")
            print(f"critic loss:\t{critic_loss}")
        if self.show_detail:
            print(f"episodic return:\t{true_return}")
            print(f"mean reward:\t{true_reward} (truth); {dream_reward} (dream)")
            print(f"actor entropy:\t{actor_entropy}")
    
    def __step_policy_model_free(self, i_step: int):
        # planing buffer
        buffer_p = Buffer(self.config, self.config.rl_args.buffer_size)
        
        # collect samples
        buffer_p.clear()
        log = self.collect(buffer_p, buffer_p.max_size, 0.)
        true_reward = log[_REWARD].mean
        true_return = log[_RETURN].mean
        self.buffer_m.append(
            buffer_p.sample_batch(self.causal_args.n_sample_collect))

        # planning
        plan_loss = self.ppo.optimize(buffer_p)
        actor_loss = plan_loss['actor'].mean
        critic_loss = plan_loss['critic'].mean
        actor_entropy = plan_loss['entropy'].mean

        # write summary
        writer = self.__writer
        writer.add_scalar('reward', true_reward, self.__n_sample)
        if not np.isnan(true_return):
            writer.add_scalar('return', true_return, self.__n_sample)
        writer.add_scalar('actor_loss', actor_loss, self.__n_sample)
        writer.add_scalar('critic_loss', critic_loss, self.__n_sample)
        writer.add_scalar('actor_entropy', actor_entropy, self.__n_sample)

        # show info
        if self.show_loss:
            print(f"actor loss:\t{plan_loss['actor'].mean}")
            print(f"critic loss:\t{plan_loss['critic'].mean}")
        if self.show_detail:
            print(f"episodic return:\t{true_return}")
            print(f"mean reward:\t{true_reward}")
            print(f"actor entropy:\t{actor_entropy}")
        if self.show_plot:
            self.ppo.show_loss(plan_loss)

    def iter_policy(self, n_step: int, model_based=False):
        for i in range(0, n_step):
            print(f"step {i}: ")
            if model_based:
                self.__step_policy_model_based(i)
                self.save_items('env-model', 'agent', 'runtime')
            else:
                self.__step_policy_model_free(i)
                self.save_items('agent', 'runtime')
            print("Done.")

    def iter_model(self, train_set: Buffer, test_set: Buffer,
                   n_step: int, n_batch_fit: int):
        writer = self.__writer

        n = len(train_set)

        # collect train samples
        self.buffer_m.clear()
        
        interval = max(n // n_step, 1)
        for i in range(0, n, interval):
            tran = train_set.transitions[i: i + interval]
            self.buffer_m.append(tran)

            # causal_reasoning
            self.causal_discovery()
            self.fit(n_batch_fit, 256)

            # eval
            log = Log()
            self.__fit_epoch(test_set, log, eval=True)

            # write summary
            writer.add_scalar('log-likeligood', -log[_NLL_LOSS].mean, self.__n_sample)
            writer.add_scalars('log_likelihood_variable',
                {k: log[_LL, k].mean for k in self.env.names_outputs}, self.__n_sample)

            print(f"test ({i}/{n}):")
            # show info
            if self.show_loss:
                print(f"- total log-likelihood:\t{-log[_NLL_LOSS].mean}")
            if self.show_detail:
                for k in self.env.names_outputs:
                    print(f"- log-likelihood of '{k}':\t{log[_LL, k].mean}")
            
            self.save_item('env-model')
