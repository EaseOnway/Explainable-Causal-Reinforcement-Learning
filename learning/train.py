import os
from typing import Any, Callable, Dict, List, Literal, Optional, final
import numpy as np
import torch
import time

import tensorboardX
from .data import Batch
from .buffer import Buffer
from .causal_discovery import discover, update
from .causal_model import CausalNet
from .config import Config
from .base import Configured
from .planning import PPO
from core import Env
from utils import Log
from utils.typings import ParentDict, NamedArrays
import utils


_LL = 'loglikelihood'
_LOSS = 'NLL loss'
_DONE = 'done'
_REWARD = 'reward'
_LOG_ROOT = './logs/'


class Train(Configured):

    def __init__(self, config: Config, name: str):
        super().__init__(config)
        print("Using following configuration:")
        print(self.config)

        self.ppo_args = self.config.ppo_args
        self.causal_args = self.config.causal_args

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
            self.outer_buffer.write(tran, reward, done)
            log[_REWARD] = reward
            log[_DONE] = done
        return log
    
    def collect_online(self):
        buffer = self.inner_buffer
        n_samples = buffer.max_size
        log = Log()
        
        buffer.clear()
        for i in range(n_samples):
            a = self.ppo.act(self.env.current_state)
            tran, reward, done, _ = self.env.step(a)
            buffer.write(tran, reward, done)
            self.outer_buffer.write(tran, reward, done)
            log[_REWARD] = reward
            log[_DONE] = done
        
        assert buffer.max_size == len(buffer)
        return log

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
    

    def update_causal_graph(self, showinfo=True):
        raise NotImplementedError
        if showinfo:
            print("---------------------confidence-----------------------")
            print(conf)

        edges_to_check = []
        for j, confj in conf.items():
            for i, confij in confj.items():
                if u.Random.event(1 - confij):
                    edges_to_check.append((i, j))

        thres = self.causal_args.pvalue_thres
        prior = self.causal_args.prior
        if self.causal_args.adaptive_thres and thres < prior:
            thres = prior - (prior - thres) * len(self.outer_buffer) / self.outer_buffer.max_size
        
        update(self.__causal_graph, self.outer_buffer, *edges_to_check,
               thres=thres, showinfo=showinfo, inplace=True)
        self.causnet.load_graph(self.__causal_graph)

        for i, j in edges_to_check:
            conf.loc[i, j] = 1

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
                                         self.causal_args.pvalue_thres,
                                         printlog)
        log_fit = self.fit(n_iter)
        return log_collect, log_fit
    
    def __get_summary_writer(self):
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
        self.config.to_txt(path + 'config.txt')

        writer = tensorboardX.SummaryWriter(path)
        return writer
        
    def run(self, n_epoch: int, 
            showinfo: Literal[None, 'brief', 'verbose', 'plot'] = 'verbose'):
        writer = self.__get_summary_writer()

        show_loss = (showinfo is not None)
        show_log_texts = (showinfo == 'verbose' or showinfo == 'plot')
        show_plot = (showinfo == 'plot')
        log_reward = Log()

        n_sample = self.causal_args.n_sample_warmup
        log_step, log_fit = self.warmup(n_sample, self.causal_args.n_iter_warmup)
        reward = log_step[_REWARD].mean

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
            log_reward(reward)

            # show running statistics
            if show_loss:
                print('')
                print(f"mean reward:\t{reward}")
                print(f"fitting loss:\t{eval[_LOSS].mean}")
            if show_log_texts:
                for k in self.env.names_outputs:
                    print(f"log-likelihood of '{k}':\t{eval[_LL, k].mean}")

            # collect new samples
            log_step = self.collect_online()
            reward = log_step[_REWARD].mean

            # update policy
            plan_loss = self.planning()
            writer.add_scalar('actor loss', plan_loss['actor'].mean, n_sample)
            writer.add_scalar('critic loss', plan_loss['critic'].mean, n_sample)
            writer.add_scalar('reward', reward, n_sample)
            
            if show_loss:
                print(f"actor loss:\t{plan_loss['actor'].mean}")
                print(f"critic loss:\t{plan_loss['critic'].mean}")

            n_sample += self.ppo_args.buffersize
            
            # update causal graph
            if not (self.ablations.graph_fixed or self.ablations.graph_offline):
                self.update_causal_graph(show_log_texts)

            # update causal equation
            fit_log = self.fit(self.causal_args.n_iter_train, eval=False)
            
            # show running statistics
            if show_plot:
                self.__show_fit_log(fit_log)
                self.ppo.show_loss(plan_loss)

            print("Done.")
        
        if show_plot:
            log_reward.plot(x = log_reward['step'].data)

        writer.close()
