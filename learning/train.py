import os
from typing import Any, Callable, Dict, List, Literal, Optional, final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
import utils as u
import utils.tensorfuncs as T
import tensorboardX

from core import Buffer, Batch
from .causal_discovery import discover, update
from .networks import CausalNet
from .networks.planning.ddpg import DDPG
from .config import Config, Configured


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



class Train(Configured):

    def __init__(self, name: str, config: Config):
        super().__init__(config)
        self.args = config.train_args
        self.causnet = CausalNet(config)
        self.buffer = Buffer(self.env.info.varinfos, self.args.buffersize)
        self.opt = config.train_args.get_optimizer(self.causnet)
        self.__causal_graph = self.init_causal_graph()

        self.planner = DDPG(config, self.causnet)
        
        self.__logdir = './logs/' + name + '/'
    
    def sample(self):
        with torch.no_grad():
            s = self.a2t(Batch.from_sample(self.env.current_state))
            a = self.planner.actor.forward(s, self.config.ddpg_args.explore_sd)
            s.update(a)
            s = self.t2a(s)
            tran, r, done, info = \
                self.env.step({k: v.squeeze(0) for k, v in s.data.items()})
            return tran, r
        

    def collect_samples(self, n: int):
        rewards = pd.Series(dtype=float)
        outcomes = pd.DataFrame(columns=self.env.names_o) # type: ignore
        for i in range(n):
            sample, reward = self.sample()
            self.buffer.write(sample)
            rewards.at[i] = reward
            outcomes.loc[i, :] = {o: float(sample[o])  # type: ignore
                                  for o in self.env.names_o}
        return outcomes.mean(axis=0), rewards.mean()

    def init_causal_graph(self):
        parent_dic = {j: [i for i in self.env.names_inputs
                          if u.basics.prob(self.args.causal_prior)]
                      for j in self.env.names_outputs}
        return parent_dic

    @final
    @property
    def causal_graph(self):
        return self.__causal_graph

    @final
    def plot_causal_graph(self, format='png'):
        return u.visualize.plot_digraph(
            self.env.names_inputs + self.env.names_outputs,
            self.__causal_graph, format=format)  # type: ignore

    class BatchInfo:
        def __init__(self, loss: float, err: pd.Series):
            self.loss = loss
            self.err = err

    @final
    def batch(self, size: int, eval=False):
        data = self.buffer.sample_batch(size)
        self.causnet.train(not eval)
        err = self.causnet.errors(self.causnet.a2t(data))
        loss = self.causnet.loss(err)

        if not eval:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        return Train.BatchInfo(float(loss),
                               pd.Series({k: float(e) for k, e in err.items()}))

    class FitInfo:
        def __init__(self, loss_batch: np.ndarray, err_batch: pd.DataFrame):
            self.loss_mean: float = np.mean(loss_batch)
            self.loss_batch = loss_batch
            self.err_mean = err_batch.mean(axis=0)
            self.err_batch = err_batch

        def show(self, errors=True, loss=True):
            if errors:
                self.err_batch.plot(alpha=0.5)
            if loss:
                plt.plot(self.loss_batch, linewidth=2, label="loss", color='k')
            plt.legend()
            plt.show()

    @final
    def fit(self, n_iter: int, eval=False):
        '''
        train network with fixed causal graph.
        '''

        losses = np.zeros(n_iter, dtype=float)
        errors = pd.DataFrame(columns=list(self.env.names_outputs))
        w = self.args.convergence_window
        for i_iter in range(n_iter):
            if self.args.check_convergence and i_iter > w:
                last_window = losses[max(0, i_iter - 2 * w): i_iter - w]
                this_window = losses[i_iter - w: i_iter]
                if np.min(last_window) < np.min(this_window):
                    losses = losses[:i_iter]
                    break

            info = self.batch(self.args.batchsize, eval)
            losses[i_iter] = info.loss
            errors.loc[i_iter, ] = info.err

        return Train.FitInfo(losses, errors)

    class PlanInfo:
        def __init__(self, actor_loss_batch: np.ndarray, critic_loss_batch: np.ndarray):
            self.actor_loss_batch = actor_loss_batch
            self.critic_loss_batch = critic_loss_batch
            self.actor_loss_mean = np.mean(actor_loss_batch)
            self.critic_loss_mean = np.mean(critic_loss_batch)

        def show(self):
            plt.plot(self.actor_loss_batch, label="actor loss")
            plt.plot(self.critic_loss_batch, label="critic loss")
            plt.legend()
            plt.show()
    
    @final
    def planning(self, n_iter: int):
        '''
        train network with fixed causal graph.
        '''

        losses_a = np.zeros(n_iter, dtype=float)
        losses_c = np.zeros(n_iter, dtype=float)

        for i_iter in range(n_iter):
            batch = self.a2t(self.buffer.sample_batch(self.args.batchsize))
            loss_a, loss_c = self.planner.train_batch(batch)
            
            losses_a[i_iter] = loss_a
            losses_c[i_iter] = loss_c

        return Train.PlanInfo(losses_a, losses_c)

    def update_causal_graph(self, conf: pd.DataFrame, showinfo=True):
        edges_to_check = []
        for j, confj in conf.items():
            for i, confij in confj.items():
                if u.basics.prob(1 - confij):
                    edges_to_check.append((i, j))

        thres = self.args.causal_pvalue_thres
        prior = self.args.causal_prior
        if self.args.adaptive_thres and thres < prior:
            thres = prior - (prior - thres) * len(self.buffer) / self.buffer.max_size
        
        update(self.__causal_graph, self.buffer, *edges_to_check,
               thres=thres, showinfo=showinfo)
        for i, j in edges_to_check:
            conf.loc[i, j] = 1

    def __eval(self, conf: pd.DataFrame):
        eval = self.batch(self.args.batchsize_eval, eval=True)
        decay = self.args.conf_decay
        prior = self.args.causal_prior
        for j in self.env.names_outputs:
            relative_err = (eval.err.loc[j]) / (eval.err.min() + 1.0)
            pas = self.__causal_graph[j]
            if len(pas) > 0:
                conf.loc[pas, j] *= np.exp(  # type: ignore
                    - decay * (len(pas) / len(self.env.names_inputs)) * (1 - prior)
                )  
            pas = list(set(self.env.names_inputs) - set(self.__causal_graph[j]))
            conf.loc[pas, j] *= np.exp(- relative_err * decay)  # type: ignore
        return eval

    class TrainInfo:
        def __init__(self, loss_epoch: np.ndarray, err_epoch: pd.DataFrame,
                     outcome_epoch: pd.DataFrame):
            self.loss_epoch = loss_epoch
            self.err_epoch = err_epoch
            self.outcome_epoch = outcome_epoch

        def show(self, errors=True, loss=True):
            if errors:
                self.err_epoch.plot(alpha=0.5)
            if loss:
                plt.plot(self.loss_epoch, linewidth=2, label="loss", color='k')
            plt.legend()
            plt.show()

            self.outcome_epoch.plot()
            plt.show()
    
    def warmup(self, n_samples, n_iter):
        self.buffer.clear()


    def run(self, n_epoch: int, 
            showinfo: Literal[None, 'brief', 'verbose', 'plot'] = 'verbose'):
        logdir = self.__logdir + "run-" + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = tensorboardX.SummaryWriter(logdir)

        conf = pd.DataFrame(
            index=self.env.names_inputs, columns=self.env.names_outputs, # type: ignore
            data=1.0) 
        conf.Name = 'confidence'

        show_loss = (showinfo is not None)
        show_log_texts = (showinfo == 'verbose' or showinfo == 'plot')
        show_plot = (showinfo == 'plot')

        
        for i in range(n_epoch):
            print(f"epoch {i}: ", end='')

            # collect samples
            outcomes, reward = self.collect_samples(self.args.num_sampling)
            writer.add_scalar('reward', reward)
            writer.add_scalars('outcome', outcomes.to_dict())  # type: ignore

            # planning with samples
            planinfo = self.planning(self.args.niter_planning_epoch)
            writer.add_scalar('actor loss', planinfo.actor_loss_mean)
            writer.add_scalar('critic loss', planinfo.critic_loss_mean)

            # load causal graph
            self.causnet.load_graph(self.__causal_graph)

            # update causal equations
            traininfo = self.fit(self.args.niter_epoch, eval=False)
            evalinfo = self.__eval(conf)
            writer.add_scalar('fitting loss', evalinfo.loss)
            writer.add_scalars('fitting error', evalinfo.err.to_dict())  # type: ignore
            
            if show_loss:
                print('')
                print(f"mean reward:\t{outcomes.at['__reward__']}")
                print(f"loss:\t{evalinfo.loss}")
            if show_log_texts:
                for k, e in evalinfo.err.items():
                    print(f"\terror of '{k}':\t{e}")
            if show_plot:
                traininfo.show()
                planinfo.show()
            
            if show_log_texts:
                print("---------------------confidence-----------------------")
                print(conf)
            self.update_causal_graph(conf, show_log_texts)

            print("Done.")
