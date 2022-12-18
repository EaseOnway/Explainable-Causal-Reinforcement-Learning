from typing import Dict, Callable, Type
from envs.lunar_lander import LunarLander
from envs.sc2_biuld_marines import SC2BuildMarine
from envs.sc2_collect import SC2Collect
from envs.taxi import Taxi
from envs.cartpole import Cartpole
from envs.cancer import Cancer
from core import Env
from learning.config import Config


ALL_ENVS: Dict[str, Type[Env]] = {
    'lunarlander': LunarLander,
    'collect': SC2Collect,
    'taxi': Taxi,
    'cartpole': Cartpole,
    'buildmarine': SC2BuildMarine,
    'cancer': Cancer,
}


def get_env_class(env_id: str):
    try:
        return ALL_ENVS[env_id]
    except KeyError as e:
        print(f"unsupported environment id '{env_id}'")
        print("supported environments are: ", sep='')
        print(', '.join(ALL_ENVS.keys()))
        raise e


def get_default_config(env_name: str):
    
    config = Config()
    config.device_id = 'cuda'
    config.rl.n_epoch_actor = 2
    config.rl.n_epoch_critic = 16
    config.rl.optim.lr = 1e-4
    config.model.pthres_independent = 0.15
    config.rl.optim.batchsize = 512
    config.model.optim.lr = 1e-4
    config.model.optim.max_grad_norm = 1.0
    config.mbrl.explore_rate_max = 0.5
    config.mbrl.n_round_planning = 20
    config.mbrl.interval_graph_update = 3
    config.mbrl.ensemble_size = 5

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
        config.mbrl.n_sample_explore = 256
        config.mbrl.n_sample_exploit = 256
        config.mbrl.n_sample_warmup = 1024
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
        config.mbrl.n_sample_warmup = 512
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
        config.mbrl.n_batch_fit = 400
        config.mbrl.n_batch_fit_new_graph = 800
        config.mbrl.n_sample_explore = 64
        config.mbrl.n_sample_exploit = 64
        config.mbrl.n_sample_warmup = 512
        config.mbrl.n_sample_rollout = 2048
        config.mbrl.rollout_length = (1, 5)
        config.model.n_jobs_fcit = 16
    elif env_name == 'lunarlander':
        config.rl.n_sample = 2048
        config.rl.discount = 0.98
        config.rl.gae_lambda = 0.975
        config.rl.kl_penalty = 0.04
        config.rl.entropy_penalty = 0.04
        config.rl.max_episode_length = 128
        config.model.buffer_size = 200000
        config.model.optim.batchsize = 1024
        config.mbrl.n_batch_fit =  400
        config.mbrl.n_batch_fit_new_graph = 800
        config.mbrl.n_sample_explore = 1024
        config.mbrl.n_sample_exploit = 1024
        config.mbrl.n_sample_warmup = 4096
        config.mbrl.n_sample_rollout = 4096
        config.mbrl.rollout_length = (1, 20)
        config.model.n_jobs_fcit = 16
    elif env_name == 'cancer':
        config.rl.n_sample = 256
        config.rl.discount = 0.95
        config.rl.gae_lambda = 0.9
        config.rl.kl_penalty = 0.2
        config.rl.entropy_penalty = 0.05
        config.rl.max_episode_length = 40
        config.model.buffer_size = 100000
        config.model.optim.batchsize = 1024
        config.mbrl.n_batch_fit =  400
        config.mbrl.n_batch_fit_new_graph = 800
        config.mbrl.n_sample_explore = 80
        config.mbrl.n_sample_exploit = 80
        config.mbrl.n_sample_warmup = 400
        config.mbrl.n_sample_rollout = 4096
        config.mbrl.rollout_length = (1, 8)
        config.model.n_jobs_fcit = 16

    return config
