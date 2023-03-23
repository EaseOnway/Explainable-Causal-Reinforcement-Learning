from typing import Dict, Callable, Type
from envs.lunar_lander import LunarLander
from envs.sc2_biuld_marines import SC2BuildMarine
from envs.cartpole import Cartpole
from envs.aimtest import AimTestEnv
from core import Env
from learning.config import Config


ALL_ENVS: Dict[str, Type[Env]] = {
    'lunarlander': LunarLander,
    'cartpole': Cartpole,
    'buildmarine': SC2BuildMarine,
    'aimtest': AimTestEnv
}


def get_env_class(env_id: str):
    try:
        return ALL_ENVS[env_id]
    except KeyError as e:
        print(f"unsupported environment id '{env_id}'")
        print("supported environments are: ", sep='')
        print(', '.join(ALL_ENVS.keys()))
        raise e


def get_default_config(env_name: str, args):
    
    config = Config()
    config.device_id = 'cuda'
    config.rl.n_epoch_actor = 2
    config.rl.n_epoch_critic = 16
    config.rl.optim.lr = 1e-4
    config.model.pthres = 0.2
    # config.model.pthres_max = 0.5
    # config.model.pthres_min = 0.15
    config.rl.optim.batchsize = 512
    config.model.optim.lr = 1e-4
    config.model.optim.max_grad_norm = 1.0
    config.mbrl.explore_rate_max = 0.8
    config.mbrl.n_round_planning = 20
    config.mbrl.causal_interval_min = 3
    config.mbrl.causal_interval_max = 3
    config.mbrl.causal_interval_increase = 0.1
    config.mbrl.ensemble_size = 5
    config.model.n_jobs_fcit = 4

    if env_name == 'cartpole':
        config.dims.variable_encoder_hidden = 64
        config.dims.variable_encoding = 64
        config.rl.n_sample = 1024
        config.rl.discount = 0.98
        config.rl.gae_lambda = 0.95
        config.rl.kl_penalty = 0.2
        config.rl.entropy_penalty = 0.04
        config.rl.max_episode_length = 200
        config.model.buffer_size = 200000
        config.model.optim.batchsize = 1024
        # config.model.n_sample_oracle = 5000
        config.mbrl.n_batch_fit =  400
        config.mbrl.n_batch_fit_new_graph = 800
        config.mbrl.n_sample_explore = 400
        config.mbrl.n_sample_exploit = 400
        config.mbrl.n_sample_warmup = 1200
        config.mbrl.n_sample_rollout = 4096
        config.mbrl.rollout_length = (1, 5)
    elif env_name == 'buildmarine':
        config.dims.variable_encoder_hidden = 64
        config.dims.variable_encoding = 64
        config.rl.n_sample = 128
        config.rl.discount = 0.95
        config.rl.gae_lambda = 0.9
        config.rl.kl_penalty = 0.2
        config.rl.entropy_penalty = 0.04
        config.rl.max_episode_length = 40
        config.model.buffer_size = 100000
        config.model.optim.batchsize = 1024
        # config.model.n_sample_oracle = 10000
        config.mbrl.n_batch_fit = 400
        config.mbrl.n_batch_fit_new_graph = 800
        config.mbrl.n_sample_explore = 64
        config.mbrl.n_sample_exploit = 64
        config.mbrl.n_sample_warmup = 512
        config.mbrl.n_sample_rollout = 2048
        config.mbrl.rollout_length = (3, 8)
    elif env_name == 'lunarlander':
        config.dims.variable_encoder_hidden = 128
        config.dims.variable_encoding = 128
        config.rl.n_sample = 2048
        config.rl.discount = 0.975
        config.rl.gae_lambda = 0.94
        config.rl.kl_penalty = 0.2
        config.rl.entropy_penalty = 0.005 if args.continuous else 0.04
        config.rl.max_episode_length = 512
        config.model.buffer_size = 200000
        config.model.optim.batchsize = 1024
        # config.model.n_sample_oracle = 200000
        config.mbrl.n_batch_fit =  400
        config.mbrl.n_batch_fit_new_graph = 800
        config.mbrl.n_sample_explore = 1024
        config.mbrl.n_sample_exploit = 1024
        config.mbrl.n_sample_warmup = 4096
        config.mbrl.n_sample_rollout = 4096
        config.mbrl.rollout_length = (1, 10)
    elif env_name == 'aimtest':
        config.dims.variable_encoder_hidden = 64
        config.dims.variable_encoding = 64
        config.dims.inferrer_key = 64
        config.dims.inferrer_value = 64
        config.dims.distribution_embedding = 32
        config.dims.decoder_hidden =32
        config.model.buffer_size = 200000
        config.model.optim.batchsize = 1024
        config.model.pthres = 0.2
        # config.model.n_sample_oracle = 200000

    return config
