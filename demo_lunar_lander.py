import torch

from envs import LunarLander
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import learning.config as cfg
import utils as u


np.set_printoptions(precision=4)


demo_env = LunarLander(render=True)


config = cfg.Config(demo_env)
config.causal_args.buffersize = 50000
config.ppo_args.buffersize = 2000
config.rl_args.discount = 0.95
config.ppo_args.gae_lambda = 0.92
config.rl_args.max_model_tr_len = 16
config.rl_args.model_ratio = 0.0
config.causal_args.pthres_independent = 0.1
config.causal_args.pthres_likeliratio = 0.1
config.device = torch.device('cpu')
config.ppo_args.entropy_penalty = 0.01
config.causal_args.optim_args.lr = 3e-4
config.ppo_args.kl_penalty = 0.1
config.causal_args.n_iter_train = 100
config.causal_args.n_iter_eval = 8
config.causal_args.optim_args.batchsize = 512
config.ppo_args.optim_args.batchsize = 512
config.ppo_args.n_epoch_actor = 8
config.ppo_args.n_epoch_critic = 32
config.ppo_args.optim_args.lr = 3e-4
# config.ablations.graph_fixed = True

trainer = learning.Train(config, "demo")

# demo
trainer.load("experiments\\LunarLander\\test\\run-1\\saved_state_dict")
demo_env = LunarLander(True)
demo_env.demo(trainer.ppo.act)
