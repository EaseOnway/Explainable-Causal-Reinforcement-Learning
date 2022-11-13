import torch

from envs import LunarLander
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import learning.config as cfg
import utils as u


np.set_printoptions(precision=4)


train_env = LunarLander()


config = cfg.Config(train_env)
config.causal_args.buffer_size = 50000
config.rl_args.buffer_size = 2048
config.rl_args.discount = 0.995
config.rl_args.gae_lambda = 0.98
config.causal_args.pthres_independent = 0.1
config.causal_args.pthres_likeliratio = 0.1
config.causal_args.maxlen_truth = 100
config.causal_args.maxlen_dream = 100
config.device = torch.device('cuda')
config.rl_args.entropy_penalty = 0.02
config.causal_args.optim_args.lr = 3e-4
config.rl_args.kl_penalty = 0.1
config.causal_args.n_iter_train = 100
config.causal_args.n_iter_eval = 8
config.causal_args.optim_args.batchsize = 512
config.rl_args.optim_args.batchsize = 512
config.rl_args.n_epoch_actor = 8
config.rl_args.n_epoch_critic = 32
config.rl_args.optim_args.lr = 3e-4

# config.ablations.no_env_model = True

trainer = learning.Train(config, "test")

trainer.init_run()
trainer.warmup(2000, 200)
trainer.iter(300)
