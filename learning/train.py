import os
from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import numpy as np
import torch
import time
from scipy.stats import chi2
import json

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
_LOG_ROOT = './logs/'
_REAL = 'real'
_MODEL = 'model'


class Train(Configured):

    def __init__(self, config: Config, name: str):
        super().__init__(config)
        print("Using following configuration:")
        print(self.config)

        self.ppo_args = self.config.ppo_args
        self.causal_args = self.config.causal_args
        self.rl_args = self.config.rl_args

        self.causnet = CausalNet(config)
        self.opt = self.F.get_optmizer(self.causal_args.optim_args, self.causnet)
        self.outer_buffer = Buffer(config, self.causal_args.buffersize)
        self.inner_buffer = Buffer(config, config.ppo_args.buffersize)
        self.causal_graph = self.init_causal_graph()
        self.ppo = PPO(config)

        self.__name = name
    

    def collect_warmup(self, n: int):
        log = Log()
        for i in range(n):
            a = self.env.random_action()
            tran, reward, done, info = self.env.step(a)
            self.outer_buffer.write(tran, reward, done, False)
            log[_REWARD] = reward
            log[_DONE] = done
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

    def init_causal_graph(self):
        parent_dic = {j: {i for i in self.env.names_inputs
                          if utils.Random.event(self.causal_args.prior)}
                      for j in self.env.names_outputs}
        return parent_dic

    @property
    @final
    def causal_graph(self):
        return self.__causal_graph

    @causal_graph.setter
    @final
    def causal_graph(self, graph: ParentDict):
        self.__causal_graph = graph
        self.causnet.load_graph(self.__causal_graph)

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
        '''
        ppo
        '''
    
        loss = self.ppo.optimize(self.inner_buffer)
        return loss

    def save_causal_graph(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__causal_graph, f)
    
    def load_causal_graph(self, path: str):
        with open(path, 'r') as f:
            causal_graph = json.load(f)
            self.causal_graph = causal_graph

    def update_causal_graph(self, eval_log: Log, best_log_probs: Dict[str, float], 
                            showinfo=True):

        # n = len(self.outer_buffer)

        to_check: List[str] = []
        n = np.sqrt(self.causal_args.optim_args.batchsize)

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
        return self.fit(self.causal_args.n_iter_eval, eval=True)
    
    def __get_data_for_causal_discovery(self) -> NamedArrays:
       temp = self.outer_buffer.tensors[:]
       temp = {k: self.raw2input(k, v).numpy() for k, v in temp.items()}
       return temp
    
    def warmup(self, n_samples, n_iter, printlog=True):
        self.outer_buffer.clear()
        log_collect = self.collect_warmup(n_samples)
        if not self.ablations.graph_fixed:
            data = self.__get_data_for_causal_discovery()
            self.causal_graph = discover(data, self.env,
                                         self.causal_args.pthres_independent,
                                         printlog)
        log_fit = self.fit(n_iter)
        return log_collect, log_fit
    
    def __make_run_dir(self):
        path = _LOG_ROOT + '/' + str(self.env) + '/' + self.__name + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        run_names = os.listdir(path)
        i = 0
        while True:
            i += 1
            run_name = "run-%d" % i
            if run_name not in run_names:
                break
        path = path + run_name + '/'

        os.makedirs(path)
        return path
        
    def run(self, n_epoch: int, 
            showinfo: Literal[None, 'brief', 'verbose', 'plot'] = 'verbose'):
        
        path = self.__make_run_dir()
        writer = tensorboardX.SummaryWriter(path)
        self.config.to_txt(path + 'config.txt')

        show_loss = (showinfo is not None)
        show_log_texts = (showinfo == 'verbose' or showinfo == 'plot')
        show_plot = (showinfo == 'plot')
        log_reward = Log()

        # warm up
        n_sample = self.causal_args.n_sample_warmup
        log_step, log_fit = self.warmup(n_sample, self.causal_args.n_iter_warmup)
        reward_real, reward_model = log_step[_REWARD].mean, np.nan
        best_log_probs = {name: -np.inf for name in self.env.names_outputs}
        
        self.save_causal_graph(path + 'causal_graph.json')

        if show_plot:
            self.__show_fit_log(log_fit)
        
        for i in range(n_epoch):
            # evaluating
            print(f"epoch {i}: ", end='')
            eval = self.__eval()
            writer.add_scalar('fitting loss', eval[_LOSS].mean, n_sample)
            writer.add_scalars(
                'log-likelihood',
                {k: eval[_LL, k].mean for k in self.env.names_outputs},
                n_sample)
            
            log_reward['step'] = n_sample
            log_reward[_REAL] = reward_real
            log_reward[_MODEL] = reward_model

            # show running statistics
            if show_loss:
                print('')
                print(f"mean reward:\t{reward_real} (real); {reward_model} (model)")
                print(f"fitting loss:\t{eval[_LOSS].mean}")
            if show_log_texts:
                for k in self.env.names_outputs:
                    print(f"log-likelihood of '{k}':\t{eval[_LL, k].mean}")

            # collect new samples
            log_step, n_sample_ = self.collect_online()
            reward_real = log_step[_REAL, _REWARD].mean
            reward_model = log_step[_MODEL, _REWARD].mean

            n_sample += n_sample_
            
            # update causal graph
            if not (self.ablations.graph_fixed or self.ablations.graph_offline):
                self.update_causal_graph(eval, best_log_probs, show_log_texts)
                self.save_causal_graph(path + 'causal_graph.json')

            # update causal equation
            fit_log = self.fit(self.causal_args.n_iter_train, eval=False)

            # update policy
            plan_loss = self.planning()
            writer.add_scalar('actor loss', plan_loss['actor'].mean, n_sample)
            writer.add_scalar('critic loss', plan_loss['critic'].mean, n_sample)
            writer.add_scalar('reward (real)', reward_real, n_sample)
            writer.add_scalar('reward (model)', reward_model, n_sample)
            
            if show_loss:
                print(f"actor loss:\t{plan_loss['actor'].mean}")
                print(f"critic loss:\t{plan_loss['critic'].mean}")
            
            # show running statistics
            if show_plot:
                self.__show_fit_log(fit_log)
                self.ppo.show_loss(plan_loss)

            print("Done.")
        
        if show_plot:
            log_reward.plot(x = log_reward['step'].data)

        writer.close()
