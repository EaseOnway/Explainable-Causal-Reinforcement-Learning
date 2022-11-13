import os
from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import numpy as np
import torch
from scipy.stats import chi2
import pickle


import tensorboardX
from .data import Batch, Transitions, Tag
from .buffer import Buffer
from .causal_discovery import discover, update
from .causal_model import CausalNet, CausalModel
from .config import Config
from .base import Configured
from .planning import PPO
from core import Env
from utils import Log
from utils.typings import ParentDict, NamedArrays, NamedTensors
import utils


_LL = 'loglikelihood'
_LOSS = 'NLL loss'
_REWARD = 'reward'
_EXP_ROOT = './experiments/'
_CAUSAL_GRAPH = 'causal.graph'
_PPO_ACTOR = 'ppo.actor'
_PPO_CRITIC = 'ppo.critic'
_CAUSAL_NET = 'causal.net'
_SAVED_STATE_DICT = 'saved_state_dict'
_SAVED_CONFIG = 'config.txt'
_RETURN = 'return'


class Train(Configured):

    def __init__(self, config: Config, name: str,
                 showinfo: Literal[None, 'brief', 'verbose', 'plot'] = 'verbose'):
        super().__init__(config)
        print("Using following configuration:")
        print(self.config)

        # arguments

        self.causal_args = self.config.causal_args
        self.rl_args = self.config.rl_args
        self.name = name
        self.dir = _EXP_ROOT + '/' + str(self.env) + '/' + self.name + '/'
        self.show_loss = (showinfo is not None)
        self.show_detail = (showinfo == 'verbose' or showinfo == 'plot')
        self.show_plot = (showinfo == 'plot')

        # causal graph, causal model, and model optimizer
        self.__causal_graph: ParentDict
        self.causnet = CausalNet(config)
        self.opt = self.F.get_optmizer(self.causal_args.optim_args, self.causnet)

        # planning algorithm
        self.ppo = PPO(config)

        # buffer for causal model
        self.buffer_m = Buffer(config, self.causal_args.buffer_size)

        # buffer for planning
        self.buffer_p = Buffer(config, config.rl_args.buffer_size)
        
        # declear runtime variables
        self.__n_sample: int
        self.__episodic_return: float
        self.__run_dir: str
        self.__writer: tensorboardX.SummaryWriter
        self.__best_log_probs: Dict[str, float]
    
    @property
    @final
    def causal_graph(self):
        return self.__causal_graph

    @causal_graph.setter
    @final
    def causal_graph(self, graph: ParentDict):
        self.__causal_graph = graph
        self.causnet.load_graph(self.__causal_graph)
    
    def state_dict(self):
        return {_PPO_ACTOR: self.ppo.actor.state_dict(),
                _PPO_CRITIC: self.ppo.critic.state_dict(),
                _CAUSAL_NET: self.causnet.state_dict(),
                _CAUSAL_GRAPH: self.__causal_graph}
    
    def load_state_dict(self, dic: Dict[str, Any]):
        self.ppo.actor.load_state_dict(dic[_PPO_ACTOR])
        self.ppo.critic.load_state_dict(dic[_PPO_CRITIC])
        self.causnet.load_state_dict(dic[_CAUSAL_NET])
        self.causal_graph = dic[_CAUSAL_GRAPH]


    def collect(self, buffer: Buffer, n_sample: int, random = False):
        '''collect real-world samples into the buffer, and compute returns'''

        log = Log()
        self.env.reset()
        initiated = True
        i_step: int = 0
        max_len = self.causal_args.maxlen_truth

        for i_sample in range(n_sample):
            # interact with the environment
            if random:
                a = self.env.random_action()
            else:
                a = self.ppo.act(self.env.current_state)
            tran, reward, terminated, _ = self.env.step(a)
            truncated = (i_step == max_len)

            # record information
            self.__episodic_return += reward
            log[_REWARD] = reward
            if truncated or terminated:
                initiated = True
                i_step = 0
                log[_RETURN] = self.__episodic_return
                self.__episodic_return = 0.
            else:
                initiated = False
                i_step += 1

            # reset environment
            if truncated:
                self.env.reset()
            
            # truncate the last sample
            if i_sample == n_sample - 1:
                truncated = True

            # write buffer
            tagcode = Tag.encode(terminated, truncated, initiated)
            buffer.write(tran, reward, tagcode)
        
        self.__n_sample += n_sample
        return log


    def dream(self, buffer: Buffer, n_sample: int):
        '''generate samples into the buffer using the environment model'''
        buffer = self.buffer_p
        log = Log()

        max_len = self.causal_args.maxlen_dream
        batchsize = self.causal_args.dream_batch_size
        env_m = CausalModel(self.causnet, self.buffer_m, batchsize, max_len)
        
        env_m.reset()
        tr: List[Transitions] = []
        
        for i_sample in range(0, n_sample, batchsize):
            with torch.no_grad():
                s = env_m.current_state
                a = self.ppo.actor.forward(s).sample().kapply(self.label2raw)
                tran = env_m.step(a)
            tr.append(tran)
            i_sample += tran.n
        
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

            t = Transitions(data, reward, code)
            buffer.append(t)
            
        assert i_sample == n_sample
        return log

    def init_params(self):
        self.ppo.actor.init_parameters()
        self.ppo.critic.init_parameters()
        self.causnet.init_parameters()
        self.causal_graph = {j: {i for i in self.env.names_inputs
                                 if utils.Random.event(self.causal_args.prior)}
                             for j in self.env.names_outputs}

    @final
    def plot_causal_graph(self, format='png'):
        return utils.visualize.plot_digraph(
            self.env.names_inputs + self.env.names_outputs,
            self.__causal_graph, format=format)  # type: ignore

    @final
    def fit_batch(self, size: int, eval=False):
        data = self.buffer_m.sample_batch(size)
        self.causnet.train(not eval)
        lls = self.causnet.get_loglikeli_dic(data)
        ll = self.causnet.loglikelihood(lls)
        loss = -ll
        
        if not eval:
            loss.backward()
            self.F.optim_step(self.causal_args.optim_args,
                              self.causnet, self.opt)

        return float(loss), {k: float(e) for k, e in lls.items()}

    @final
    def fit(self, n_iter: int, eval=False):
        '''
        train network with fixed causal graph.
        '''
        args = self.causal_args.optim_args
        log = Log()

        for i_iter in range(n_iter):
            loss, lls = self.fit_batch(args.batchsize, eval)
            log[_LOSS] = loss
            for k, ll in lls.items():
                log[_LL, k] = ll

        return log

    def __show_fit_log(self, log: Log):
        Log.figure(figsize=(12, 5))  # type: ignore
        Log.subplot(121, title=_LOSS)
        log[_LOSS].plot(color='k')
        Log.subplot(122, title=_LL)
        log[_LL].plots(self.env.names_outputs)
        Log.show()
    
    @final
    def planning(self):
        loss = self.ppo.optimize(self.buffer_p)
        self.__writer.add_scalar('actor loss', loss['actor'].mean, self.__n_sample)
        self.__writer.add_scalar('critic loss', loss['critic'].mean, self.__n_sample)
        return loss
        
    def load(self, path: Optional[str] = None):
        path = path or self.__run_dir + _SAVED_STATE_DICT
        self.load_state_dict(torch.load(path))

    def save(self, path: Optional[str] = None):
        path = path or self.__run_dir + _SAVED_STATE_DICT
        torch.save(self.state_dict(), path)

    def __update_causal_graph(self, eval_log: Log,
                              best_log_probs: Dict[str, float], 
                              showinfo=True):

        # n = len(self.outer_buffer)

        to_check: List[str] = []
        n = np.log2(self.causal_args.optim_args.batchsize)

        for name in self.env.names_outputs:
            eval_log_prob = eval_log[_LL, name].mean

            if eval_log_prob >= best_log_probs[name]:
                best_log_probs[name] = eval_log_prob
                if showinfo:
                    print(f"log-p of {name} increased to {eval_log_prob}")
                continue

            likeli_ratio = 2 * n * (best_log_probs[name] - eval_log_prob)
            p_value = 1 - chi2.cdf(likeli_ratio, len(self.env.names_inputs))

            if showinfo:
                print(f"likelihood ratio {name} = {likeli_ratio}, "
                      f"raising p_value = {p_value}")

            if p_value < self.causal_args.pthres_likeliratio:
                print(f"reject causal structure of {name}.")
                best_log_probs[name] = -np.inf
                to_check.append(name)

        thres = self.causal_args.pthres_independent
        prior = self.causal_args.prior
        if self.causal_args.adaptive_thres and thres < prior:
            thres = prior - (prior - thres) * len(self.buffer_m)\
                / self.buffer_m.max_size

        if len(to_check) > 0:
            print(f"start causal discovery for {', '.join(to_check)}.")
            data = self.__get_data_for_causal_discovery()
            for target in to_check:
                update(self.__causal_graph, data, self.env, target,
                    thres=thres, showinfo=showinfo, inplace=True)
            self.causnet.load_graph(self.__causal_graph)

    def __eval(self):
        eval = self.fit(self.causal_args.n_iter_eval, eval=True)
        writer = self.__writer
        writer.add_scalar('fitting loss', eval[_LOSS].mean, self.__n_sample)
        writer.add_scalars(
            'log-likelihood',
            {k: eval[_LL, k].mean for k in self.env.names_outputs},
            self.__n_sample)
        return eval
    
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
        self.config.to_txt(path + _SAVED_CONFIG)

        if resume:
            if n_sample is None:
                raise ValueError
            self.load()
            self.__n_sample = n_sample
            self.__writer = tensorboardX.SummaryWriter(path, purge_step=n_sample)
        else:
            self.__n_sample = 0
            self.init_params()
            self.save()
            self.__writer = tensorboardX.SummaryWriter(path)

        self.__episodic_return = 0.
        self.__best_log_probs = {name: -np.inf for name in self.env.names_outputs}
        
        self.env.reset()
        self.buffer_m.clear()
        self.buffer_p.clear()

    def warmup(self, n_sample: int, n_iter: int):
        log_collect = self.collect(self.buffer_m, n_sample, random=True)

        print(f"warm up: ", end='')

        # causal discovery
        if not self.ablations.graph_fixed:
            data = self.__get_data_for_causal_discovery()
            self.causal_graph = discover(data, self.env,
                                         self.causal_args.pthres_independent,
                                         self.show_detail)
        
        # fit causal equation
        fit_log = self.fit(n_iter)
        eval = self.__eval()
        self.save()
        self.__best_log_probs = {k: eval[_LL, k].mean
                                 for k in self.env.names_outputs}

        # show info
        if self.show_loss:
            print('')
            print(f"fitting loss:\t{eval[_LOSS].mean}")
        if self.show_detail:
            for k in self.env.names_outputs:
                print(f"log-likelihood of '{k}':\t{eval[_LL, k].mean}")
        if self.show_plot:
            self.__show_fit_log(fit_log)
        
        print('Done.')
    
        return log_collect, fit_log
    
    def __iter_step(self):
        # collect true samples
        log_step = self.collect(self.buffer_m, self.causal_args.n_truth)
        true_reward = log_step[_REWARD].mean
        true_return = log_step[_RETURN].mean

        # fit causal equation
        fit_log = self.fit(self.causal_args.n_iter_train, eval=False)

        # evaluate causal model
        eval = self.__eval()

        # dream samples
        self.buffer_p.clear()
        dream_log = self.dream(self.buffer_p, self.buffer_p.max_size)
        dream_reward = dream_log[_REWARD].mean

        # planning
        plan_loss = self.planning()
        actor_loss = plan_loss['actor'].mean
        critic_loss = plan_loss['critic'].mean

        # update causal graph
        if not (self.ablations.graph_fixed or self.ablations.graph_offline):
            self.__update_causal_graph(eval, self.__best_log_probs, self.show_detail)

        # write summary
        writer = self.__writer
        writer.add_scalar('reward (truth)', true_reward, self.__n_sample)
        if not np.isnan(true_return):
            writer.add_scalar('return', true_return, self.__n_sample)
        writer.add_scalar('reward (dream)', dream_reward, self.__n_sample)
        writer.add_scalar('actor loss', actor_loss, self.__n_sample)
        writer.add_scalar('critic loss', critic_loss, self.__n_sample)

        # show info
        if self.show_loss:
            print(f"mean reward:\t{true_reward} (truth); {dream_reward} (dream)")
            print(f"episodic return:\t{true_return}")
            print(f"actor loss:\t{plan_loss['actor'].mean}")
            print(f"critic loss:\t{plan_loss['critic'].mean}")
            print(f"fitting loss:\t{eval[_LOSS].mean}")
        if self.show_detail:
            for k in self.env.names_outputs:
                print(f"log-likelihood of '{k}':\t{eval[_LL, k].mean}")
        if self.show_plot:
            self.__show_fit_log(fit_log)
            self.ppo.show_loss(plan_loss)
    
    def __iter_step_no_env_model(self):
        # collect samples
        self.buffer_p.clear()
        log = self.collect(self.buffer_p, self.buffer_p.max_size)
        true_reward = log[_REWARD].mean
        true_return = log[_RETURN].mean

        # planning
        plan_loss = self.planning()
        actor_loss = plan_loss['actor'].mean
        critic_loss = plan_loss['critic'].mean

        # write summary
        writer = self.__writer
        writer.add_scalar('reward (real)', true_reward, self.__n_sample)
        if not np.isnan(true_return):
            writer.add_scalar('return', true_return, self.__n_sample)
        writer.add_scalar('actor loss', actor_loss, self.__n_sample)
        writer.add_scalar('critic loss', critic_loss, self.__n_sample)

        # show info
        if self.show_loss:
            print(f"mean reward:\t{true_reward} (truth)")
            print(f"episodic return:\t{true_return}")
            print(f"actor loss:\t{plan_loss['actor'].mean}")
            print(f"critic loss:\t{plan_loss['critic'].mean}")
        if self.show_plot:
            self.ppo.show_loss(plan_loss)

    def iter(self, n_epoch: int):
        for i in range(n_epoch):
            print(f"epoch {i}: ")
            if self.ablations.no_env_model:
                self.__iter_step_no_env_model()
            else:
                self.__iter_step()
            self.save()
            print("Done.")
