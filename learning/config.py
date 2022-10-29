from typing import Any, Dict, Literal, Optional

import torch
import numpy as np
import utils.tensorfuncs as T
from core import Env, Batch


class _BaseConfig:
    def _override(self, **kargs):
        for k, v in vars(self).items():
            if k in kargs:
                v_ = kargs[k]
                if not isinstance(v_, type(v)):
                    raise TypeError(
                        f"the type of argument '{k}' should be {type(v)}, but recieved {type(v_)}")
                print(f"override parameter: {k} = {v_}")
                super().__setattr__(k, v_)


class NetDims(_BaseConfig):
    def __init__(self, **kargs):
        self.action_encoder_hidden: int = 8
        self.state_encoder_hidden: int = 8
        self.action_encoding: int = 8
        self.state_encoding: int = 16
        self.inferrer_value: int = 16
        self.inferrer_key: int = 16
        self.action_aggregator_hidden: int = 16
        self.inferrer_decoder_hidden: int = 32
        self.actor_hidden: int = 64

        self._override(**kargs)


class Ablations(_BaseConfig):
    def __init__(self, **kargs):
        self.no_attn = False
        self.recur = False

        self._override(**kargs)


class TorchArgs(_BaseConfig):
    def __init__(self, **kargs):
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self._override(**kargs)

        self.dic = {'device': self.device, 'dtype': self.dtype}


class DDPGArgs(_BaseConfig):
    def __init__(self, **kargs):
        self.gamma = 0.98
        self.target_update_rate = 0.1
        self.explore_sd = 1.0

        self._override(**kargs)


class TrainArgs(_BaseConfig):
    def __init__(self, **kargs):
        self.buffersize = 10000
        self.batchsize = 128
        self.n_iter_epoch = 10
        self.n_iter_planning = 10
        self.check_convergence = False
        self.convergence_window = 10
        self.optalg: Literal["Adam", "SGD"] = "Adam"
        self.optargs: Dict[str, Any] = {}
        self.lr = 0.01
        self.causal_prior = 0.25
        self.causal_pvalue_thres = 0.05
        self.adaptive_thres = True
        self.batchsize_eval = 128
        self.conf_decay = 0.1
        self.n_sample_epoch = 500
        self.n_sample_warmup = 2000
        self.n_iter_warmup = 50

        self._override(**kargs)

    def get_optimizer(self, network: torch.nn.Module) -> torch.optim.Optimizer:
        if self.optalg == "Adam":
            return torch.optim.Adam(network.parameters(), self.lr, **self.optargs)
        elif self.optalg == "SGD":
            return torch.optim.SGD(network.parameters(), self.lr, **self.optargs)
        else:
            raise ValueError(f"unsupported algorithm: {self.optalg}")


class Config(_BaseConfig):
    def __init__(self, env: Env, **kargs):
        self.dims = NetDims(**kargs)
        self.torch_args = TorchArgs(**kargs)
        self.ablations = Ablations(**kargs)
        self.train_args = TrainArgs(**kargs)
        self.ddpg_args = DDPGArgs(**kargs)
        self._override(**kargs)

        self.env = env

class Configured:
    def __init__(self, config: Config):
        self.__config = config
        self.__env = config.env

    @property
    def config(self):
        return self.__config

    @property
    def dims(self):
        return self.__config.dims

    @property
    def torchargs(self):
        return self.__config.torch_args.dic
    
    @property
    def env(self):
        return self.__env

    @property
    def ablations(self):
        return self.__config.ablations

    def v(self, name: str):
        return self.__config.env.var(name)

    def a2t(self, d: Batch[np.ndarray]) -> Batch[torch.Tensor]:
        return Batch(d.n, {k: T.a2t(a, **self.torchargs)
                           for k, a in d.items()})

    def t2a(self, d: Batch[torch.Tensor]) -> Batch[np.ndarray]:
        return Batch(d.n, {k: T.t2a(t, self.v(k).dtype)
                           for k, t in d.items()})

    def step_shift(self, d: Batch[torch.Tensor]):
        return Batch.torch(d.n, {name: d[self.env.name_next_step(name)]
                                 for name in self.env.names_s})

    def get_outcome_vectors(self, d: Batch[torch.Tensor]):
        return T.batch_flatcat([d[o] for o in self.env.names_o], d.n)

    def get_outcome_weights(self):
        return T.a2t(self.env.weights_o, **self.torchargs)

    def batch_rewards(self, d: Batch[torch.Tensor]):
        outcomes = self.get_outcome_vectors(d)
        return outcomes @ self.get_outcome_weights()
