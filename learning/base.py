from typing import Dict, final, Optional, Any

import torch.nn as nn
import torch
import torch.distributions as D

from .config import *
from core import Batch, Distributions, Transitions
from core import VType, DType

import utils as u
from utils.typings import *


class Functional:

    @staticmethod
    def get_optmizer(args: OptimArgs, network: torch.nn.Module):
        algorithm = args.algorithm
        lr = args.lr
        kargs = args.alg_args
        if algorithm == "Adam":
            return torch.optim.Adam(network.parameters(), lr, **kargs)
        elif algorithm == "SGD":
            return torch.optim.SGD(network.parameters(), lr, **kargs)
        else:
            raise ValueError(f"unsupported algorithm: {algorithm}")
    
    @staticmethod
    def optim_step(args: OptimArgs, network: torch.nn.Module,
                   opt: torch.optim.Optimizer):
        if args.use_grad_clip:
            torch.nn.utils.clip_grad.clip_grad_norm_(
                network.parameters(), args.max_grad_norm)
        opt.step()
        opt.zero_grad()

class Configured:

    F = Functional

    def __init__(self, config: Config):
        if not config.__readonly__:
            config.confirm()
            print("Configuration Confirmed!")

        self.__config = config
        self.__env = config.env
        self.__device = config.device
        self.__torchargs = {'device': self.__device,
                            'dtype': DType.Real.torch}
        self.__tensor_operator = u.TensorOperator(**self.__torchargs)
    
    @property
    @final
    def config(self):
        return self.__config

    @property
    @final
    def dims(self):
        return self.__config.dims

    @property
    @final
    def torchargs(self):
        return self.__torchargs
    
    @property
    @final
    def device(self):
        return self.__device
    
    @property
    @final
    def env(self):
        return self.__env

    @property
    @final
    def ablations(self):
        return self.__config.ablations
    
    @property
    @final
    def T(self):
        return self.__tensor_operator

    @final
    def v(self, name: str):
        return self.__config.env.var(name)

    @final
    def shift_states(self, transition: Batch):
        return Batch(transition.n,
            {s: transition[s_] for s, s_ in self.env.nametuples_s})
    
    @final
    def named_tensors(self, kvalues: NamedValues, device: Optional[Any] = None)\
            -> NamedTensors:
        device = device or self.__device
        return {k: self.v(k).tensor(v, device) for k, v in kvalues.items()}

    def raw2input(self, name: str, raw: torch.Tensor):
        return self.v(name).raw2input(raw)
    
    def raw2label(self, name: str,  raw: torch.Tensor):
        return self.v(name).raw2label(raw)
    
    def label2raw(self, name: str, label: torch.Tensor):
        return self.v(name).label2raw(label)
    
    def as_numpy(self, raw: Batch, drop_batch=True):
        if drop_batch and raw.n == 1:
            return {k: self.T.t2a(v.squeeze(0), self.v(k).dtype.numpy)
                    for k, v in raw.items()}
        else:
            return {k: self.T.t2a(v, self.v(k).dtype.numpy)
                    for k, v in raw.items()}



class BaseNN(nn.Module, Configured):

    def __init__(self, config: Config):
        nn.Module.__init__(self)
        Configured.__init__(self, config)

    def init_parameters(self):
        for p in self.parameters():
            if p.ndim < 2:
                nn.init.normal_(p)
            else:
                nn.init.orthogonal_(p)
