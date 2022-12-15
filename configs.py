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
    config.rl.n_round_model_based = 25
    config.rl.optim.lr = 1e-4
    config.env_model.pthres_independent = 0.15
    config.rl.optim.batchsize = 512
    config.env_model.optim.lr = 1e-4
    config.env_model.optim.max_grad_norm = 1.0
    config.env_model.explore_rate_max = 0.3

    if env_name == 'cartpole':
        config.rl.buffer_size = 2048 if model_based else 1024
        config.rl.discount = 0.98
        config.rl.gae_lambda = 0.95
        config.rl.kl_penalty = 0.1
        config.rl.entropy_penalty = 0.04
        config.env_model.buffer_size = 200000
        config.env_model.maxlen_truth = 128
        config.env_model.maxlen_dream = 128
        config.env_model.n_batch_fit =  512
        config.env_model.n_batch_fit_new_graph = 2048
        config.env_model.optim.batchsize = 1024
        config.env_model.n_sample_explore = 512
        config.env_model.n_sample_exploit = 512
        config.env_model.interval_graph_update = 16
        config.env_model.n_jobs_fcit = 16
        config.env_model.n_ensemble = 1
    elif env_name == 'collect':
        config.rl.buffer_size = 2048 if model_based else 120
        config.rl.discount = 0.975
        config.rl.gae_lambda = 0.94
        config.rl.kl_penalty = 0.1
        config.rl.entropy_penalty = 0.04
        config.env_model.buffer_size = 200000
        config.env_model.maxlen_truth = 40
        config.env_model.maxlen_dream = 40
        config.env_model.n_batch_fit =  2048
        config.env_model.n_batch_fit_new_graph = 4096
        config.env_model.optim.batchsize = 1024
        config.env_model.n_sample_explore = 60
        config.env_model.n_sample_exploit = 60
        config.env_model.interval_graph_update = 16
        config.env_model.n_jobs_fcit = 16
        config.env_model.n_ensemble = 3
    elif env_name == 'buildmarine':
        config.rl.buffer_size = 2048 if model_based else 120
        config.rl.discount = 0.95
        config.rl.gae_lambda = 0.9
        config.rl.kl_penalty = 0.1
        config.rl.entropy_penalty = 0.04
        config.env_model.buffer_size = 100000
        config.env_model.maxlen_truth = 40
        config.env_model.maxlen_dream = 40
        config.env_model.n_batch_fit =  2048
        config.env_model.n_batch_fit_new_graph = 4096
        config.env_model.optim.batchsize = 1024
        config.env_model.n_sample_explore = 60
        config.env_model.n_sample_exploit = 60
        config.env_model.interval_graph_update = 8
        config.env_model.n_jobs_fcit = 16
        config.env_model.n_ensemble = 3
    elif env_name == 'lunarlander':
        config.rl.buffer_size = 4096 if model_based else 2048
        config.rl.discount = 0.975
        config.rl.gae_lambda = 0.94
        config.rl.kl_penalty = 0.1
        config.rl.entropy_penalty = 0.02
        config.env_model.buffer_size = 200000
        config.env_model.maxlen_truth = 1024
        config.env_model.maxlen_dream = 1024
        config.env_model.n_batch_fit =  512
        config.env_model.n_batch_fit_new_graph = 2048
        config.env_model.optim.batchsize = 1024
        config.env_model.n_sample_explore = 1024
        config.env_model.n_sample_exploit = 1024
        config.env_model.interval_graph_update = 16
        config.env_model.n_jobs_fcit = 16
        config.env_model.n_ensemble = 1
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
