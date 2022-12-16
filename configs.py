from typing import Literal, Optional
from learning.config import Config
from torch import device
import envs


N_WARM_UP = {
    'cartpole':  2048,
    'buildmarine': 512,
    'collect': 512,
    'lunarlander': 4096,
}


def make_config(env_name: str, model_based: bool,
                ablation: Optional[str] = None):
    
    config = Config()
    config.device_id = 'cuda'
    config.env_id = env_name
    config.rl.n_epoch_actor = 2 if model_based else 8
    config.rl.n_epoch_critic = 16 if model_based else 64
    config.rl.optim.lr = 1e-4
    config.model.pthres_independent = 0.15
    config.rl.optim.batchsize = 512
    config.model.optim.lr = 1e-4
    config.model.optim.max_grad_norm = 1.0
    config.mbrl.explore_rate_max = 0.5
    config.mbrl.n_round_planning = 20
    config.mbrl.interval_graph_update = 3
    config.mbrl.ensemble_size = 5 if model_based else 1

    if env_name == 'cartpole':
        config.rl.n_sample = 1024
        config.rl.discount = 0.98
        config.rl.gae_lambda = 0.95
        config.rl.kl_penalty = 0.1
        config.rl.entropy_penalty = 0.04
        config.rl.max_episode_length = 128
        config.model.buffer_size = 200000
        config.model.optim.batchsize = 1024
        config.mbrl.n_batch_fit =  400
        config.mbrl.n_batch_fit_new_graph = 800
        config.mbrl.n_sample_explore = 512
        config.mbrl.n_sample_exploit = 512
        config.mbrl.n_sample_rollout = 4096
        config.mbrl.rollout_length = (1, 20)
        config.model.n_jobs_fcit = 16
    elif env_name == 'collect':
        config.rl.n_sample = 128
        config.rl.discount = 0.95
        config.rl.gae_lambda = 0.9
        config.rl.kl_penalty = 0.1
        config.rl.entropy_penalty = 0.04
        config.rl.max_episode_length = 128
        config.model.buffer_size = 100000
        config.model.optim.batchsize = 1024
        config.mbrl.n_batch_fit =  400
        config.mbrl.n_batch_fit_new_graph = 800
        config.mbrl.n_sample_explore = 64
        config.mbrl.n_sample_exploit = 64
        config.mbrl.n_sample_rollout = 2048
        config.mbrl.rollout_length = (1, 5)
        config.model.n_jobs_fcit = 16
    elif env_name == 'buildmarine':
        config.rl.n_sample = 128
        config.rl.discount = 0.95
        config.rl.gae_lambda = 0.9
        config.rl.kl_penalty = 0.1
        config.rl.entropy_penalty = 0.04
        config.rl.max_episode_length = 128
        config.model.buffer_size = 100000
        config.model.optim.batchsize = 1024
        config.mbrl.n_batch_fit =  512
        config.mbrl.n_batch_fit_new_graph = 2048
        config.mbrl.n_sample_explore = 64
        config.mbrl.n_sample_exploit = 64
        config.mbrl.n_sample_rollout = 2048
        config.mbrl.rollout_length = (1, 5)
        config.model.n_jobs_fcit = 16
    elif env_name == 'lunarlander':
        config.rl.n_sample = 2048
        config.rl.discount = 0.98
        config.rl.gae_lambda = 0.975
        config.rl.kl_penalty = 0.94
        config.rl.entropy_penalty = 0.04
        config.rl.max_episode_length = 128
        config.model.buffer_size = 200000
        config.model.optim.batchsize = 1024
        config.mbrl.n_batch_fit =  512
        config.mbrl.n_batch_fit_new_graph = 2048
        config.mbrl.n_sample_explore = 1024
        config.mbrl.n_sample_exploit = 1024
        config.mbrl.n_sample_rollout = 4096
        config.mbrl.rollout_length = (1, 20)
        config.model.n_jobs_fcit = 16
    else:
        raise ValueError(f"unknown environment: {env_name}")

    if ablation == 'no_attn':
        config.ablations.no_attn = True
    elif ablation == 'recur':
        config.ablations.recur = True
    elif ablation == 'offline':
        config.ablations.offline = True
    elif ablation == 'dense':
        config.ablations.dense = True
    elif ablation == 'mlp':
        config.ablations.mlp = True
    elif ablation is not None:
        raise NotImplementedError("Ablation not supported")

    return config
