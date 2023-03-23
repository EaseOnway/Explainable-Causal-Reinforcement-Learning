from typing import Any, Dict, Literal, Set, Tuple, final, Optional, Union
import torch

import json


class _BaseConfig:

    @final
    def __lines(self, parent: str = ""):
        lines = []
        for k, v in self.__dict__.items():
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
    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, _BaseConfig):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out
    
    @final
    def load_dict(self, d: Dict):
        for k, v in d.items():
            try:
                temp = getattr(self, k)
            except AttributeError:
                print(f"config warning: can not parse '{k}'")
                continue
            if isinstance(temp, _BaseConfig):
                temp.load_dict(v)
            else:
                setattr(self, k, v)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, path):
        with open(path, 'r') as f:
            self.load_dict(json.load(f))


class NetDims(_BaseConfig):
    def __init__(self):
        self.variable_encoding: int = 64
        self.variable_encoder_hidden: int = 64
        self.action_influce_embedding: int = 128
        self.aggregator_hidden: int = 64
        self.inferrer_value: int = 128
        self.inferrer_key: int = 64
        self.distribution_embedding: int = 128
        self.decoder_hidden: int = 128
        self.mlp_model_hidden: int = 256
        self.actor_critic_hidden: int = 64


class Ablations(_BaseConfig):
    def __init__(self):
        self.recur = False
        self.mlp = False
        self.offline = False
        self.dense = False
        self.no_attn = False

class OptimArgs(_BaseConfig):
    def __init__(self):
        self.lr = 1e-4
        self.algorithm = "AdamW"
        self.alg_args: Dict[str, Any] = {}
        self.batchsize = 128
        self.use_grad_clip = True
        self.max_grad_norm = 1.0


class RLArgs(_BaseConfig):
    def __init__(self):
        self.discount = 0.95  # gamma
        self.gae_lambda = 0.9
        self.kl_penalty = 0.1
        self.entropy_penalty = 0.01
        self.n_epoch_critic = 5
        self.n_epoch_actor = 1
        self.optim = OptimArgs()
        self.use_adv_norm = True
        self.max_episode_length: Optional[int] = 100
        self.use_reward_scaling = True
        self.n_sample = 1000


class ModelArgs(_BaseConfig):
    def __init__(self):
        self.buffer_size = 10000  
        self.optim = OptimArgs()
        self.prior = 0.25
        self.pthres = 0.15
        self.n_jobs_fcit = 1
        
class MBRLArgs(_BaseConfig):
    def __init__(self):
        self.rollout_length: Union[int, Tuple[int, int]] = 1
        self.n_sample_rollout = 100
        self.n_sample_explore = 100
        self.n_sample_exploit = 100
        self.n_sample_warmup = 400
        self.dream_batch_size = 64
        self.explore_rate_max = 0.5  # exploration rate for estimating model
        self.causal_interval_min: int = 1
        self.causal_interval_max: int = 12
        self.causal_interval_increase: Union[float, int] = 1
        self.n_round_planning = 25
        self.n_batch_fit = 50
        self.n_batch_fit_new_graph = 500
        self.ensemble_size = 1

class Baseline(_BaseConfig):
    def __init__(self):
        self.sparse_factor = 0.001
        self.optim = OptimArgs()
        self.dim_q_hidden = 128
        self.n_sample_importance = 12

class Config(_BaseConfig):
    def __init__(self):
        self.dims = NetDims()
        self.ablations = Ablations()
        self.model = ModelArgs()
        self.rl = RLArgs()
        self.mbrl = MBRLArgs() 
        self.device_id: str = 'cpu'
        self.baseline = Baseline()
