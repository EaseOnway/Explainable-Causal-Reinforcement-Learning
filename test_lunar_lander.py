import torch

from envs import LunarLander
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import pandas as pd
import learning.config as cfg
import utils as u


np.set_printoptions(precision=4)


env = LunarLander()

# mdp.scm.plot().view('./causal_graph.gv')
# env.demo()

config = cfg.Config(env)
config.causal_args.buffersize = 20000
config.ppo_args.buffersize = 2000
config.rl_args.discount = 0.9
config.rl_args.model_ratio = 0.75
config.rl_args.max_model_tr_len = 8
config.causal_args.n_sample_warmup = 2000
config.causal_args.pthres_independent = 0.1
config.causal_args.pthres_likeliratio = 0.1
config.device = torch.device('cuda')
config.ppo_args.gae_lambda = 0.9
config.ppo_args.entropy_penalty = 0.01
config.causal_args.optim_args.lr = 0.001
config.ppo_args.kl_penalty = 0.1
config.causal_args.n_iter_train = 100
config.causal_args.n_iter_eval = 8
config.causal_args.optim_args.batchsize = 512
config.ppo_args.optim_args.batchsize = 512
config.ppo_args.n_epoch_actor = 5
config.ppo_args.n_epoch_critic = 20
config.ppo_args.optim_args.lr = 0.001
# config.ablations.graph_fixed = True


trainer = learning.Train(config, "test")
trainer.run(50, 'verbose')
