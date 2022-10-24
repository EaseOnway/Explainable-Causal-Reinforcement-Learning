import os
from typing import Any, Callable, Dict, List, Literal, Optional, final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import utils
from core import Buffer

from .causal_discovery import discover, update
from .networks import CausalNet
from .networks.config import NetConfig

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TrainConfig:
    def __init__(self, netcfg: NetConfig, buffersize=10000, batchsize=128,
                 niter_epoch=10, abort_window: Optional[int] = None,
                 lr=0.01, optalg: Literal["Adam", "SGD"] = "Adam", optargs: Dict = {},
                 causal_prior=0.2, causal_pvalue_thres=0.1, adaptive_thres=True,
                 batchsize_eval=128, conf_decay=0.1, num_sampling=500):
        self.netcfg = netcfg
        self.buffersize = buffersize
        self.batchsize = batchsize
        self.niter_epoch = niter_epoch
        self.abort_window = abort_window
        self.optalg = optalg
        self.optargs = optargs
        self.lr = lr
        self.causal_prior = causal_prior
        self.causal_pvalue_thres = causal_pvalue_thres
        self.adaptive_thres = adaptive_thres
        self.batchsize_eval = batchsize_eval
        self.conf_decay = conf_decay
        self.num_sampling = num_sampling

    def get_optimizer(self, network: torch.nn.Module) -> torch.optim.Optimizer:
        if self.optalg == "Adam":
            return torch.optim.Adam(network.parameters(), self.lr, **self.optargs)
        elif self.optalg == "SGD":
            return torch.optim.SGD(network.parameters(), self.lr, **self.optargs)
        else:
            raise ValueError(f"unsupported algorithm: {self.optalg}")


class Train:
    Config = TrainConfig

    def __init__(self, config: TrainConfig,
                 sampler: Callable[[], Dict[str, Any]]):
        self.config = config
        self.network = CausalNet(config.netcfg)
        self.task = config.netcfg.task
        self.opt = config.get_optimizer(self.network)
        self.buffer = Buffer(config.netcfg.task.varinfos, config.buffersize)
        self.sampler = sampler
        self.__causal_graph = self.init_causal_graph()

    def collect_samples(self, n: int):
        for _ in range(n):
            self.buffer.write(**self.sampler())

    def init_causal_graph(self):
        parent_dic = {j: [i
                          for i in self.inkeys if utils.prob(self.config.causal_prior)
                          ] for j in self.outkeys}
        return parent_dic

    @final
    @property
    def outkeys(self):
        return self.config.netcfg.outkeys

    @final
    @property
    def inkeys(self):
        return self.config.netcfg.inkeys

    @final
    @property
    def causal_graph(self):
        return self.__causal_graph

    @final
    def plot_causal_graph(self):
        return utils.plot_digraph(self.inkeys | self.outkeys,
                                  self.__causal_graph)  # type: ignore

    class BatchInfo:
        def __init__(self, loss: float, err: pd.Series):
            self.loss = loss
            self.err = err

    @final
    def batch(self, size: int, eval=False):
        data = self.buffer.sample_batch(size)
        self.network.train(not eval)
        err = self.network.errors(data)
        loss = self.network.loss(err)

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
    def fit(self, n_iter: int, eval=False,
            abort_window: Optional[int] = None):
        '''
        train network with fixed causal graph.
        '''

        losses = np.zeros(n_iter, dtype=float)
        errors = pd.DataFrame(columns=list(self.outkeys))

        for i_iter in range(n_iter):
            if abort_window is not None and i_iter > abort_window:
                last_window = losses[max(0, i_iter - 2 * abort_window):
                                     i_iter - abort_window]
                this_window = losses[i_iter - abort_window: i_iter]
                if np.min(last_window) < np.min(this_window):
                    losses = losses[:i_iter]
                    break

            info = self.batch(self.config.batchsize, eval)
            losses[i_iter] = info.loss
            errors.loc[i_iter, ] = info.err

        return Train.FitInfo(losses, errors)

    def update_causal_graph(self, conf: pd.DataFrame, showinfo=True):
        edges_to_check = []
        for j, confj in conf.items():
            for i, confij in confj.items():
                if utils.prob(1 - confij):
                    edges_to_check.append((i, j))

        thres = self.config.causal_pvalue_thres
        prior = self.config.causal_prior
        if self.config.adaptive_thres and thres < prior:
            thres = prior - (prior - thres) * len(self.buffer) / self.buffer.max_size
        
        update(self.__causal_graph, self.buffer, *edges_to_check,
               thres=self.config.causal_pvalue_thres, showinfo=showinfo)
        for i, j in edges_to_check:
            conf.loc[i, j] = 1

    def __eval(self, conf: pd.DataFrame):
        eval = self.batch(self.config.batchsize_eval, eval=True)
        decay = self.config.conf_decay
        prior = self.config.causal_prior
        for j in self.outkeys:
            relative_err = (eval.err.loc[j]) / (eval.err.min() + 1.0)
            pas = self.__causal_graph[j]
            if len(pas) > 0:
                conf.loc[pas, j] *= np.exp(  # type: ignore
                    - decay * (len(pas) / len(self.inkeys)) * (1 - prior)
                )  
            pas = list(self.inkeys - set(self.__causal_graph[j]))
            conf.loc[pas, j] *= np.exp(  # type: ignore
                - relative_err * self.config.conf_decay)
        return eval

    class TrainInfo:
        def __init__(self, loss_epoch: np.ndarray, err_epoch: pd.DataFrame):
            self.loss_epoch = loss_epoch
            self.err_epoch = err_epoch

        def show(self, errors=True, loss=True):
            if errors:
                self.err_epoch.plot(alpha=0.5)
            if loss:
                plt.plot(self.loss_epoch, linewidth=2, label="loss", color='k')
            plt.legend()
            plt.show()

    def run(self, n_epoch: int, 
            showinfo: Literal[None, 'brief', 'verbose', 'plot'] = 'verbose'):
        conf = pd.DataFrame(index=list(self.inkeys), columns=list(self.outkeys),
                            data=1.0)
        conf.Name = 'confidence'
        error_log = pd.DataFrame(columns=list(self.outkeys))
        loss_lis = []

        show_loss = (showinfo is not None)
        show_log_texts = (showinfo == 'verbose' or showinfo == 'plot')
        show_plot = (showinfo == 'plot')
        
        for i in range(n_epoch):
            print(f"epoch {i}: ", end='')
            self.network.load_graph(self.__causal_graph)
            self.collect_samples(self.config.num_sampling)
            traininfo = self.fit(self.config.niter_epoch, eval=False,
                                 abort_window=self.config.abort_window)
            evalinfo = self.__eval(conf)
            loss_lis.append(evalinfo.loss)
            error_log.loc[i, ] = evalinfo.err
            

            if show_loss:
                print('')
                print(f"loss:\t{evalinfo.loss}")
            if show_log_texts:
                for k, e in evalinfo.err.items():
                    print(f"\terror of '{k}':\t{e}")
                print("---------------------confidence-----------------------")
                print(conf)
            if show_plot:
                traininfo.show()
                
            self.update_causal_graph(conf, show_log_texts)
            print("Done.")

        return Train.TrainInfo(np.array(loss_lis), error_log)
