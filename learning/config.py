from typing import Any, Dict, Literal, Set, Tuple, final, Optional
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
            temp = getattr(self, k)
            if isinstance(temp, _BaseConfig):
                temp.load_dict(v)
            else:
                setattr(self, k, v)
    
    def save(self, path: str):
        if path[-5:] != '.json':
            path = path + '.json'
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, path: str):
        if path[-5:] != '.json':
            path = path + '.json'
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
        self.inferrer_feed_forward: int = 128
        self.decoder_hidden: int = 128
        self.mlp_model_hidden: int = 256
        self.actor_critic_hidden: int = 64


class Ablations(_BaseConfig):
    def __init__(self):
        self.no_attn = False
        self.recur = False
        self.mlp = False
        self.offline = False
        self.dense = False

class OptimArgs(_BaseConfig):
    def __init__(self):
        self.lr = 1e-4
        self.algorithm = "Adam"
        self.alg_args: Dict[str, Any] = {}
        self.batchsize = 128
        self.use_grad_clip = True
        self.max_grad_norm = 1.0


class RLArgs(_BaseConfig):
    def __init__(self):
        self.buffer_size = 2000
        self.discount = 0.95  # gamma
        self.gae_lambda = 0.9
        self.kl_penalty = 0.1
        self.entropy_penalty = 0.01
        self.n_epoch_critic = 5
        self.n_epoch_actor = 1
        self.n_round_model_based = 20
        self.optim = OptimArgs()
        self.use_adv_norm = True
        self.use_reward_scaling = True


class EnvModelArgs(_BaseConfig):
    def __init__(self):
        self.buffer_size = 10000  
        self.maxlen_truth: Optional[int] = 100
        self.maxlen_dream: Optional[int] = 100
        self.dream_batch_size = 32
        self.explore_rate_max = 0.2  # exploration rate for estimating model
        self.n_sample_explore = 100
        self.n_sample_exploit = 100
        self.n_batch_fit = 50
        self.n_batch_fit_new_graph = 500
        self.interval_graph_update = 50
        self.optim = OptimArgs()
        self.prior = 0.25
        self.pthres_independent = 0.05
        self.n_jobs_fcit = -1
        self.n_ensemble = 1


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
        self.env_model = EnvModelArgs()
        self.rl = RLArgs()
        self.device_id: str = 'cpu'
        self.env_id: str = 'UNDEFINED'
        self.baseline = Baseline()
