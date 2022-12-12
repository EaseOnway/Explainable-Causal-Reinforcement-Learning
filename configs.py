from typing import Literal
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
                model_based: bool):
    
    config = Config()
    config.device = device('cuda')
    config.rl_args.n_epoch_actor = 2 if model_based else 8
    config.rl_args.n_epoch_critic = 16 if model_based else 64
    config.rl_args.n_round_model_based = 25
    config.rl_args.optim_args.lr = 1e-4
    config.causal_args.pthres_independent = 0.15
    config.rl_args.optim_args.batchsize = 512
    config.causal_args.optim_args.lr = 1e-4
    config.causal_args.optim_args.max_grad_norm = 10
    config.causal_args.explore_rate_max = 0.3

    if env_name == 'cartpole':
        config.env = envs.Cartpole()
        config.rl_args.buffer_size = 2048 if model_based else 512
        config.rl_args.discount = 0.98
        config.rl_args.gae_lambda = 0.95
        config.rl_args.kl_penalty = 0.1
        config.rl_args.entropy_penalty = 0.04
        config.causal_args.buffer_size = 200000
        config.causal_args.maxlen_truth = 128
        config.causal_args.maxlen_dream = 128
        config.causal_args.n_batch_fit =  512
        config.causal_args.n_batch_fit_new_graph = 2048
        config.causal_args.optim_args.batchsize = 1024
        config.causal_args.n_sample_collect = 512
        config.causal_args.n_sample_evaluate_policy = 512
        config.causal_args.interval_graph_update = 16
        config.causal_args.n_jobs_fcit = 16
        config.causal_args.n_ensemble = 1
    elif env_name == 'collect':
        config.env = envs.SC2Collect()
        config.rl_args.buffer_size = 2048 if model_based else 80
        config.rl_args.discount = 0.975
        config.rl_args.gae_lambda = 0.94
        config.rl_args.kl_penalty = 0.1
        config.rl_args.entropy_penalty = 0.04
        config.causal_args.buffer_size = 200000
        config.causal_args.maxlen_truth = 40
        config.causal_args.maxlen_dream = 40
        config.causal_args.n_batch_fit =  512
        config.causal_args.n_batch_fit_new_graph = 2048
        config.causal_args.optim_args.batchsize = 1024
        config.causal_args.n_sample_collect = 80
        config.causal_args.n_sample_evaluate_policy = 40
        config.causal_args.interval_graph_update = 16
        config.causal_args.n_jobs_fcit = 16
        config.causal_args.n_ensemble = 1
    elif env_name == 'buildmarine':
        config.env = envs.SC2BuildMarine()
        config.rl_args.buffer_size = 2048 if model_based else 80
        config.rl_args.discount = 0.95
        config.rl_args.gae_lambda = 0.9
        config.rl_args.kl_penalty = 0.1
        config.rl_args.entropy_penalty = 0.04
        config.causal_args.buffer_size = 100000
        config.causal_args.maxlen_truth = 40
        config.causal_args.maxlen_dream = 40
        config.causal_args.n_batch_fit =  512
        config.causal_args.n_batch_fit_new_graph = 2048
        config.causal_args.optim_args.batchsize = 1024
        config.causal_args.n_sample_collect = 80
        config.causal_args.n_sample_evaluate_policy = 40
        config.causal_args.interval_graph_update = 8
        config.causal_args.n_jobs_fcit = 16
        config.causal_args.n_ensemble = 1
    elif env_name == 'lunarlander':
        config.env = envs.LunarLander(continuous=True)
        config.rl_args.buffer_size = 4096 if model_based else 1024
        config.rl_args.discount = 0.975
        config.rl_args.gae_lambda = 0.94
        config.rl_args.kl_penalty = 0.1
        config.rl_args.entropy_penalty = 0.002
        config.causal_args.buffer_size = 200000
        config.causal_args.maxlen_truth = 1024
        config.causal_args.maxlen_dream = 1024
        config.causal_args.n_batch_fit =  512
        config.causal_args.n_batch_fit_new_graph = 2048
        config.causal_args.optim_args.batchsize = 1024
        config.causal_args.n_sample_collect = 1024
        config.causal_args.n_sample_evaluate_policy = 1024
        config.causal_args.interval_graph_update = 16
        config.causal_args.n_jobs_fcit = 16
        config.causal_args.n_ensemble = 1
    else:
        raise ValueError(f"unknown environment: {env_name}")

    return config
