from typing import Any, Dict, Literal, Set, Tuple, final
from core import Env
import torch


class _BaseConfig:
    __reserved = ("__readonly__")

    def __init__(self):
        self.__readonly__: bool
        super().__setattr__("__readonly__", False)
    
    def check_valid(self):
        assert True
    
    @final
    def confirm(self):
        try:
            self.check_valid()
        except Exception as e:
            print("Invalid Configuration!")
            raise e

        super().__setattr__("__readonly__", True)
        for k, v in self.items():
            if isinstance(v, _BaseConfig):
                v.confirm()

    @final
    def items(self):
        for k, v in vars(self).items():
            if k not in _BaseConfig.__reserved:
                yield k, v

    @final
    def __lines(self, parent: str = ""):
        lines = []
        for k, v in self.items():
            if isinstance(v, _BaseConfig):
                lines.extend(v.__lines(parent + k + '.'))
            else:
                lines.append(parent + k + ': ' + str(v))
        return lines
    
    @final
    def __str__(self):
        return '\n'.join(self.__lines())

    @final
    def __repr__(self):
        return str(self)
    
    @final
    def to_txt(self, file: str):
        if file[-4:] != '.txt':
            file = file + '.txt'
        
        with open(file, 'w') as f:
            f.write(str(self))
    
    @final
    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in _BaseConfig.__reserved:
            raise ValueError(f"Cannot set reserver attribute {__name}")
        elif self.__readonly__:
            raise ValueError(f"Cannot modify read-only configuration.")
        else:
            super().__setattr__(__name, __value)



class NetDims(_BaseConfig):
    def __init__(self):
        super().__init__()

        self.action_encoder_hidden: int = 8
        self.state_encoder_hidden: int = 8
        self.action_encoding: int = 8
        self.state_encoding: int = 16
        self.inferrer_value: int = 16
        self.inferrer_key: int = 16
        self.action_aggregator_hidden: int = 16
        self.inferrer_feed_forward: int = 32
        self.decoder_hidden: int = 32
        self.actor_critic_hidden: int = 64


class Ablations(_BaseConfig):
    def __init__(self):
        super().__init__()

        self.no_attn = False
        self.recur = False
        self.graph_fixed = False  # graph never updates
        self.graph_offline = False  # graph only updates in warmup


class OptimArgs(_BaseConfig):
    def __init__(self):
        super().__init__()

        self.lr = 0.01
        self.algorithm = "Adam"
        self.alg_args: Dict[str, Any] = {}
        self.batchsize = 128
        self.use_grad_clip = True
        self.max_grad_norm = 1.0

class PPOArgs(_BaseConfig):
    def __init__(self):
        super().__init__()

        self.gamma = 0.98
        self.gae_lambda = 0.9
        self.kl_penalty = 0.1
        self.entropy_penalty = 0.01
        self.n_epoch_critic = 5
        self.n_epoch_actor = 1
        self.buffersize = 1000
        self.optim_args = OptimArgs()

class CausalArgs(_BaseConfig):
    def __init__(self):
        super().__init__()

        self.buffersize = 10000
        self.n_iter_train = 10
        self.n_iter_eval = 2
        self.optim_args = OptimArgs()
        self.prior = 0.25
        self.pvalue_thres = 0.05
        self.conf_decay = 0.1
        self.adaptive_thres = True
        self.n_sample_warmup = 2000
        self.n_iter_warmup = 50

class Config(_BaseConfig):
    def __init__(self, env: Env):
        super().__init__()
        self.dims = NetDims()
        self.ablations = Ablations()
        self.causal_args = CausalArgs()
        self.ppo_args = PPOArgs()
        self.env = env
        self.device = torch.device('cpu')
