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


def make_config(env_name: Literal['cartpole',
                                  'collect', 
                                  'buildmarine',
                                  'lunarlander'],
                model_based: bool,
                ablation: Optional[str] = None):
    
    config = Config()
    config.device = device('cuda')
    config.rl_args.n_epoch_actor = 2 if model_based else 8
    config.rl_args.n_epoch_critic = 16 if model_based else 64
    config.rl_args.n_round_model_based = 25
    config.rl_args.optim_args.lr = 1e-4
    config.envmodel_args.pthres_independent = 0.15
    config.rl_args.optim_args.batchsize = 512
    config.envmodel_args.optim_args.lr = 1e-4
    config.envmodel_args.optim_args.max_grad_norm = 1.0
    config.envmodel_args.explore_rate_max = 0.3

    if env_name == 'cartpole':
        config.env = envs.Cartpole()
        config.rl_args.buffer_size = 2048 if model_based else 512
        config.rl_args.discount = 0.98
        config.rl_args.gae_lambda = 0.95
        config.rl_args.kl_penalty = 0.1
        config.rl_args.entropy_penalty = 0.04
        config.envmodel_args.buffer_size = 200000
        config.envmodel_args.maxlen_truth = 128
        config.envmodel_args.maxlen_dream = 128
        config.envmodel_args.n_batch_fit =  512
        config.envmodel_args.n_batch_fit_new_graph = 2048
        config.envmodel_args.optim_args.batchsize = 1024
        config.envmodel_args.n_sample_collect = 512
        config.envmodel_args.n_sample_evaluate_policy = 512
        config.envmodel_args.interval_graph_update = 16
        config.envmodel_args.n_jobs_fcit = 16
        config.envmodel_args.n_ensemble = 1
        config.baseline.buffersize = 10000
    elif env_name == 'collect':
        config.env = envs.SC2Collect()
        config.rl_args.buffer_size = 2048 if model_based else 80
        config.rl_args.discount = 0.975
        config.rl_args.gae_lambda = 0.94
        config.rl_args.kl_penalty = 0.1
        config.rl_args.entropy_penalty = 0.04
        config.envmodel_args.buffer_size = 200000
        config.envmodel_args.maxlen_truth = 40
        config.envmodel_args.maxlen_dream = 40
        config.envmodel_args.n_batch_fit =  512
        config.envmodel_args.n_batch_fit_new_graph = 2048
        config.envmodel_args.optim_args.batchsize = 1024
        config.envmodel_args.n_sample_collect = 80
        config.envmodel_args.n_sample_evaluate_policy = 40
        config.envmodel_args.interval_graph_update = 16
        config.envmodel_args.n_jobs_fcit = 16
        config.envmodel_args.n_ensemble = 1
        config.baseline.buffersize = 2000
    elif env_name == 'buildmarine':
        config.env = envs.SC2BuildMarine()
        config.rl_args.buffer_size = 2048 if model_based else 80
        config.rl_args.discount = 0.95
        config.rl_args.gae_lambda = 0.9
        config.rl_args.kl_penalty = 0.1
        config.rl_args.entropy_penalty = 0.04
        config.envmodel_args.buffer_size = 100000
        config.envmodel_args.maxlen_truth = 40
        config.envmodel_args.maxlen_dream = 40
        config.envmodel_args.n_batch_fit =  512
        config.envmodel_args.n_batch_fit_new_graph = 2048
        config.envmodel_args.optim_args.batchsize = 1024
        config.envmodel_args.n_sample_collect = 80
        config.envmodel_args.n_sample_evaluate_policy = 40
        config.envmodel_args.interval_graph_update = 8
        config.envmodel_args.n_jobs_fcit = 16
        config.envmodel_args.n_ensemble = 1
        config.baseline.buffersize = 2000
    elif env_name == 'lunarlander':
        config.env = envs.LunarLander(continuous=True)
        config.rl_args.buffer_size = 4096 if model_based else 2048
        config.rl_args.discount = 0.975
        config.rl_args.gae_lambda = 0.94
        config.rl_args.kl_penalty = 0.1
        config.rl_args.entropy_penalty = 0.002
        config.envmodel_args.buffer_size = 200000
        config.envmodel_args.maxlen_truth = 1024
        config.envmodel_args.maxlen_dream = 1024
        config.envmodel_args.n_batch_fit =  512
        config.envmodel_args.n_batch_fit_new_graph = 2048
        config.envmodel_args.optim_args.batchsize = 1024
        config.envmodel_args.n_sample_collect = 1024
        config.envmodel_args.n_sample_evaluate_policy = 1024
        config.envmodel_args.interval_graph_update = 16
        config.envmodel_args.n_jobs_fcit = 16
        config.envmodel_args.n_ensemble = 1
        config.baseline.buffersize = 20000
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
