import torch

from envs import SC2BuildMarine
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import learning.config as cfg
import utils as u
from absl import app

from learning import Explainner


np.set_printoptions(precision=4)



config = cfg.Config()
config.causal_args.buffer_size = 100000
config.rl_args.buffer_size = 512
config.rl_args.discount = 0.95
config.rl_args.gae_lambda = 0.9
config.causal_args.pthres_independent = 0.2
config.causal_args.pthres_likeliratio = 0.1
config.causal_args.maxlen_truth = 100
config.causal_args.maxlen_dream = 100
config.device = torch.device('cuda')
config.rl_args.entropy_penalty = 0.03
config.causal_args.optim_args.lr = 1e-4
config.rl_args.kl_penalty = 0.1
config.causal_args.n_epoch_fit =  50
config.causal_args.n_epoch_fit_new_graph = 200
config.causal_args.optim_args.batchsize = 512
config.causal_args.n_true_sample = 256
config.causal_args.interval_graph_update = 8
config.rl_args.optim_args.batchsize = 512
config.rl_args.n_epoch_actor = 2
config.rl_args.n_epoch_critic = 16
config.rl_args.n_round_model_based = 50
config.rl_args.optim_args.lr = 1e-4


def train_model_based(args):
    config.env = SC2BuildMarine()
    config.rl_args.buffer_size = 1024

    trainer = learning.Train(config, "test", 'verbose')
    trainer.init_run()
    trainer.warmup(512, random=True)
    trainer.iter_policy(300, model_based=True)
    trainer.causal_reasoning(300)


def train_model_free(args):
    config.env = SC2BuildMarine()
    config.rl_args.buffer_size = 256

    trainer = learning.Train(config, "test", 'verbose')
    trainer.init_run()
    trainer.iter_policy(300, model_based=False)


def causal_resoning(args):
    config.env = SC2BuildMarine()
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run("experiments/LunarLander/test/run-2",
                     resume=True)
    trainer.warmup(10000, random=True)
    trainer.warmup(50000)
    trainer.causal_reasoning(300)
    trainer.save()


def explain(args):
    raise NotImplementedError
    config.env = SC2BuildMarine()
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run("experiments/LunarLander/test/exemplar",
                     resume=True)
    
    trainer.plot_causal_graph().view()
    exp = Explainner(trainer)
    trainer.warmup(100)
    tran = trainer.buffer_m.arrays[50]
    a = trainer.env.random_action()
    exp.whynot(trainer.env.state_of(tran), a,
               mode=True, thres=0.15, maxlen=20)


if __name__ == "__main__":
    app.run(train_model_based)