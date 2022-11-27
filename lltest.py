import torch

from envs import LunarLander

import learning
import numpy as np
import learning.config as cfg
from absl import app

from learning import Explainner


np.set_printoptions(precision=4)


def make_config(model_based: bool):
    config = cfg.Config()
    config.env = LunarLander(continuous=True)
    config.device = torch.device('cuda')
    config.rl_args.buffer_size = 4096
    config.rl_args.discount = 0.98
    config.rl_args.gae_lambda = 0.94
    config.rl_args.kl_penalty = 0.1
    config.rl_args.entropy_penalty = 0.02
    config.rl_args.optim_args.batchsize = 1024
    config.rl_args.n_epoch_actor = 2 if model_based else 4
    config.rl_args.n_epoch_critic = 16 if model_based else 32
    config.rl_args.n_round_model_based = 50
    config.rl_args.optim_args.lr = 1e-4
    config.causal_args.buffer_size = 200000
    config.causal_args.pthres_independent = 0.1
    config.causal_args.pthres_likeliratio = 0.1
    config.causal_args.maxlen_truth = 100
    config.causal_args.maxlen_dream = 100
    config.causal_args.optim_args.lr = 1e-4
    config.causal_args.n_epoch_fit =  50
    config.causal_args.n_epoch_fit_new_graph = 200
    config.causal_args.optim_args.batchsize = 1024
    config.causal_args.n_true_sample = 1024
    config.causal_args.interval_graph_update = 8
    config.causal_args.n_jobs_fcit = 4
    return config


def train_model_based(_):
    config = make_config(model_based=True)
    trainer = learning.Train(config, "model_based", 'verbose')
    trainer.init_run("experiments/SC2BuildMarine/model_based/run-1")
    trainer.warmup(4096, random=True)
    trainer.iter_policy(300, model_based=True)


def train_model_free(_):
    config = make_config(model_based=False)
    trainer = learning.Train(config, "model_free", 'verbose')
    trainer.init_run()
    trainer.iter_policy(300, model_based=False)


def causal_resoning(_):
    config = make_config(model_based=True)
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run("experiments/LunarLander/test/run-2",
                     resume=True)
    trainer.warmup(10000, random=True)
    trainer.warmup(50000)
    trainer.causal_reasoning(300)
    trainer.save()


def explain(_):
    config = make_config(model_based=True)
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run("experiments/SC2BuildMarine/test/run-1",
                     resume=True)
    
    trainer.plot_causal_graph().view()
    exp = Explainner(trainer)
    trainer.warmup(15)
    tran = trainer.buffer_m.arrays[12]
    a = trainer.env.action_of(tran)

    exp.why(trainer.env.state_of(tran),
            {'build_barracks': a['build_barracks']},
            mode=True, thres=0.2, maxlen=10)

    exp.whynot(trainer.env.state_of(tran), trainer.env.random_action(),
               mode=True, thres=0.15, maxlen=10)


if __name__ == "__main__":
    learning.Train.set_seed(1)
    app.run(train_model_based)
    # app.run(explain)
