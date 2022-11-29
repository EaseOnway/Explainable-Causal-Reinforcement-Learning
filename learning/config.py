from typing import Any, Dict, Literal, Set, Tuple, final, Optional
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

        self.action_encoder_hidden: int = 64
        self.state_encoder_hidden: int = 64
        self.action_encoding: int = 64
        self.state_encoding: int = 64
        self.inferrer_value: int = 128
        self.inferrer_key: int = 64
        self.action_aggregator_hidden: int = 64
        self.inferrer_feed_forward: int = 128
        self.decoder_hidden: int = 128
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

        self.lr = 1e-3
        self.algorithm = "Adam"
        self.alg_args: Dict[str, Any] = {}
        self.batchsize = 128
        self.use_grad_clip = True
        self.max_grad_norm = 1.0


class RLArgs(_BaseConfig):
    def __init__(self):
        super().__init__()
        self.buffer_size = 2000
        self.discount = 0.95  # gamma
        self.gae_lambda = 0.9
        self.kl_penalty = 0.1
        self.entropy_penalty = 0.01
        self.n_epoch_critic = 5
        self.n_epoch_actor = 1
        self.n_round_model_based = 20
        self.optim_args = OptimArgs()
        self.use_adv_norm = True
        self.use_reward_scaling = True


class CausalArgs(_BaseConfig):
    def __init__(self):
        super().__init__()
        self.buffer_size = 10000  
        self.maxlen_truth: Optional[int] = 100
        self.maxlen_dream: Optional[int] = 100
        self.dream_batch_size = 32
        self.n_true_sample = 200
        self.n_batch_fit = 50
        self.n_batch_fit_new_graph = 500
        self.interval_graph_update = 50
        self.optim_args = OptimArgs()
        self.prior = 0.25
        self.pthres_independent = 0.05
        self.pthres_likeliratio = 0.1
        self.adaptive_thres = True
        self.n_jobs_fcit = -1
        self.n_ensemble = 1


class Config(_BaseConfig):
    def __init__(self):
        super().__init__()
        self.dims = NetDims()
        self.ablations = Ablations()
        self.causal_args = CausalArgs()
        self.rl_args = RLArgs()
        self.device = torch.device('cpu')
        self.env: Env
    
    def check_valid(self):
        if not hasattr(self, 'env'):
            raise AttributeError(
                "missing configuration of 'env' (environment)")
