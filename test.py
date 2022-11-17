import torch

from envs import LunarLander
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import learning.config as cfg
import utils as u

from learning.action_effect import Explainner


np.set_printoptions(precision=4)


train_env = LunarLander()


config = cfg.Config(train_env)
config.causal_args.buffer_size = 50000
config.rl_args.buffer_size = 4096
config.rl_args.discount = 0.98
config.rl_args.gae_lambda = 0.94
config.causal_args.pthres_independent = 0.1
config.causal_args.pthres_likeliratio = 0.1
config.causal_args.maxlen_truth = 1024
config.causal_args.maxlen_dream = 1024
config.device = torch.device('cuda')
config.rl_args.entropy_penalty = 0.02
config.causal_args.optim_args.lr = 1e-4
config.rl_args.kl_penalty = 0.1
config.causal_args.n_epoch_each_step =  100
config.causal_args.optim_args.batchsize = 512
config.causal_args.n_truth = 1024
config.rl_args.optim_args.batchsize = 512
config.rl_args.n_epoch_actor = 2
config.rl_args.n_epoch_critic = 16
config.rl_args.optim_args.lr = 1e-4


if False:
    trainer = learning.Train(config, "test", 'verbose')
    trainer.init_run()
    trainer.iter_policy(300, model_based=False)

if False:
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run("experiments/LunarLander/test/exemplar",
                     resume=True)
    trainer.warmup(10000, random=True)
    trainer.warmup(20000)
    # trainer.causal_reasoning(100)
    trainer.causnet.init_parameters()
    trainer.fit(300)
    trainer.save()

if True:
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run("experiments/LunarLander/test/exemplar",
                     resume=True)
    
    #trainer.plot_causal_graph().view()
    exp = Explainner(trainer)
    trainer.warmup(100)
    tran = trainer.buffer_m.arrays[50]

    exp.explain(tran, mode=True, thres=0.2, maxlen=50)
