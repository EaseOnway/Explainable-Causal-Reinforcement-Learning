import os
from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import numpy as np
import torch
from scipy.stats import chi2
import pickle


import tensorboardX
from .data import Batch, Transitions
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
_DONE = 'done'
_REWARD = 'reward'
_EXP_ROOT = './experiments/'
_CAUSAL_GRAPH = 'causal.graph'
_PPO_ACTOR = 'ppo.actor'
_PPO_CRITIC = 'ppo.critic'
_CAUSAL_NET = 'causal.net'
_SAVED_STATE_DICT = 'saved_state_dict'
_SAVED_CONFIG = 'config.txt'
_REAL = 'real'
_MODEL = 'model'
_RETURN = 'return'


class Train(Configured):

    def __init__(self, config: Config, name: str,
                 showinfo: Literal[None, 'brief', 'verbose', 'plot'] = 'verbose'):
        super().__init__(config)
        print("Using following configuration:")
        print(self.config)

        self.ppo_args = self.config.ppo_args
        self.causal_args = self.config.causal_args
        self.rl_args = self.config.rl_args

        self.causnet = CausalNet(config)
        self.ppo = PPO(config)

        self.opt = self.F.get_optmizer(self.causal_args.optim_args, self.causnet)
        self.outer_buffer = Buffer(config, self.causal_args.buffersize)
        self.inner_buffer = Buffer(config, config.ppo_args.buffersize)
        
        self.name = name
        self.dir = _EXP_ROOT + '/' + str(self.env) + '/' + self.name + '/'

        self.show_loss = (showinfo is not None)
        self.show_detail = (showinfo == 'verbose' or showinfo == 'plot')
        self.show_plot = (showinfo == 'plot')

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

    def collect_warmup(self, n: int):
        log = Log()
        for i in range(n):
            a = self.env.random_action()
            tran, reward, done, info = self.env.step(a)
            self.outer_buffer.write(tran, reward, done, False)
            
            log[_REWARD] = reward
            log[_DONE] = done

            self.__episodic_return += reward
            if done:
                log[_RETURN] = self.__episodic_return
                self.__episodic_return = 0.

        return log

    def collect_online(self):
        buffer = self.inner_buffer
        n_model = int(self.ppo_args.buffersize * self.rl_args.model_ratio)
        n_real = buffer.max_size - n_model
        log = Log()
        
        buffer.clear()

        for i in range(n_real):
            a = self.ppo.act(self.env.current_state)
            tran, reward, done, _ = self.env.step(a)
            truncated = (i == n_real - 1)
            buffer.write(tran, reward, done, truncated)
            self.outer_buffer.write(tran, reward, done, truncated)
            
            log[_REAL, _REWARD] = reward

            self.__episodic_return += reward
            if done:
                log[_RETURN] = self.__episodic_return
                self.__episodic_return = 0.

        env_m = CausalModel(self.causnet, self.inner_buffer,
                            self.rl_args.model_batchsize)
        
        while len(buffer) < buffer.max_size:
            tr: List[Transitions] = []
            env_m.reset()
            
            with torch.no_grad():
                for i in range(self.rl_args.max_model_tr_len):
                    s = env_m.current_state
                    a = self.ppo.actor.forward(s).sample().\
                        kapply(self.label2raw)
                    tran = env_m.step(a)
                    tr.append(tran)
            
            for i in range(self.rl_args.model_batchsize):
                if len(buffer) >= buffer.max_size:
                    break

                rj = range(min(len(tr), buffer.max_size-len(buffer)))
                data = {name: torch.stack([tr[j][name][i] for j in rj])
                        for name in self.env.names_all}
                reward = torch.stack([tr[j].rewards[i]  for j in rj])
                code = torch.stack([tr[j].code[i] for j in rj])

                for r in reward:
                    log[_MODEL, _REWARD] = float(r)

                if code[-1] != Transitions.DONE:
                    code[-1] = Transitions.TRUNCATED

                t = Transitions(data, reward, code)
                buffer.append(t)
            
        assert buffer.max_size == len(buffer)
        return log, n_real

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
        data = self.outer_buffer.sample_batch(size)
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
        loss = self.ppo.optimize(self.inner_buffer)
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
            thres = prior - (prior - thres) * len(self.outer_buffer)\
                / self.outer_buffer.max_size

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
       temp = self.outer_buffer.tensors[:]
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
        self.outer_buffer.clear()
        self.inner_buffer.clear()

    def warmup(self, n_sample: int, n_iter: int):
        log_collect = self.collect_warmup(n_sample)
        self.__n_sample += n_sample

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

    def iter(self, n_epoch: int):
        
        # perpareation
        writer = self.__writer

        # start iteration
        for i in range(n_epoch):
            # evaluating
            print(f"epoch {i}: ", end='')

            # collect new samples
            log_step, n = self.collect_online()
            reward_real = log_step[_REAL, _REWARD].mean
            reward_model = log_step[_MODEL, _REWARD].mean
            return_ = log_step[_RETURN].mean
            writer.add_scalar('reward (real)', reward_real, self.__n_sample)
            writer.add_scalar('reward (model)', reward_model, self.__n_sample)
            if not np.isnan(return_):
                writer.add_scalar('return', return_, self.__n_sample)
            
            self.__n_sample = self.__n_sample + n

            # fit causal equation
            fit_log = self.fit(self.causal_args.n_iter_train, eval=False)

            # evaluate causal model
            eval = self.__eval()
            
            # update causal graph
            if not (self.ablations.graph_fixed or self.ablations.graph_offline):
                self.__update_causal_graph(eval, self.__best_log_probs, self.show_detail)

            # update policy
            plan_loss = self.planning()
            writer.add_scalar('actor loss', plan_loss['actor'].mean, self.__n_sample)
            writer.add_scalar('critic loss', plan_loss['critic'].mean, self.__n_sample)

            # show info
            if self.show_loss:
                print('')
                print(f"mean reward:\t{reward_real} (real); {reward_model} (model)")
                print(f"actor loss:\t{plan_loss['actor'].mean}")
                print(f"critic loss:\t{plan_loss['critic'].mean}")
                print(f"episodic return:\t{return_}")
                print(f"fitting loss:\t{eval[_LOSS].mean}")
            if self.show_detail:
                for k in self.env.names_outputs:
                    print(f"log-likelihood of '{k}':\t{eval[_LL, k].mean}")
            if self.show_plot:
                self.__show_fit_log(fit_log)
                self.ppo.show_loss(plan_loss)
            
            self.save()
            print("Done.")
