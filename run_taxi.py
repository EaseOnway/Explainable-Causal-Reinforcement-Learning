import torch
import numpy as np
from envs import Taxi
import learning
import learning.config as cfg
from learning import Explainner

np.set_printoptions(precision=4)


dir_ = None
ablation = None


def make_config(model_based: bool):
    config = cfg.Config()
    config.env = Taxi()
    config.device = torch.device('cuda')
    config.rl_args.buffer_size = 2048
    config.rl_args.discount = 0.95
    config.rl_args.gae_lambda = 0.9
    config.rl_args.kl_penalty = 0.1
    config.rl_args.entropy_penalty = 0.04
    config.rl_args.optim_args.batchsize = 512
    config.rl_args.n_epoch_actor = 2 if model_based else 8
    config.rl_args.n_epoch_critic = 16 if model_based else 64
    config.rl_args.n_round_model_based = 25
    config.rl_args.optim_args.lr = 1e-4
    config.causal_args.buffer_size = 100000
    config.causal_args.pthres_independent = 0.15
    config.causal_args.pthres_likeliratio = 0.1
    config.causal_args.maxlen_truth = 128
    config.causal_args.maxlen_dream = 128
    config.causal_args.optim_args.lr = 1e-4
    config.causal_args.optim_args.max_grad_norm = 10
    config.causal_args.n_batch_fit =  512
    config.causal_args.n_batch_fit_new_graph = 2048
    config.causal_args.optim_args.batchsize = 1024
    config.causal_args.n_true_sample = 2048
    config.causal_args.interval_graph_update = 8
    config.causal_args.n_jobs_fcit = 16
    config.causal_args.n_ensemble = 3 if model_based else 1

    return config


def train_model_based(_):
    config = make_config(model_based=True)
    
    if ablation is not None:
        expname = 'model_based_' + ablation
    else:
        expname = 'model_based'
    
    if ablation == 'no_attn':
        config.ablations.no_attn = True
    elif ablation == 'recur':
        config.ablations.recur = True
    elif ablation == 'offline':
        config.ablations.offline = True
    elif ablation is not None:
        raise NotImplementedError("Ablation not supported")
    
    trainer = learning.Train(config, expname, 'verbose')
    trainer.init_run(dir_)
    trainer.warmup(2048, random=True)
    trainer.iter_policy(300, model_based=True)


def train_model_free(_):
    config = make_config(model_based=False)
    trainer = learning.Train(config, "model_free", 'verbose')
    trainer.init_run(dir_)
    trainer.iter_policy(300, model_based=False)
    trainer.causal_reasoning(2048)


def causal_resoning(_):
    config = make_config(model_based=True)
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run(dir_, resume=True)
    trainer.warmup(1000, random=True)
    trainer.warmup(2000)
    trainer.causal_reasoning(300)
    trainer.save()


def explain(_):
    config = make_config(model_based=True)
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run(dir_, resume=True)
    trainer.plot_causal_graph().view()
    exp = Explainner(trainer)
    trainer.warmup(15)
    tran = trainer.buffer_m.arrays[12]
    a = trainer.env.action_of(tran)

    exp.why(trainer.env.state_of(tran), a,
            mode=True, thres=0.2, maxlen=10, complete=True)

    exp.whynot(trainer.env.state_of(tran), trainer.env.random_action(),
               mode=True, thres=0.15, maxlen=10)


if __name__ == "__main__":
    from absl import app
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('command', type=str, default='model_based',
                        help="'model_based', 'model_free', or 'explain'")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--ablation', type=str, default=None)
    args = parser.parse_args()

    if args.seed is not None:
      learning.Train.set_seed(args.seed)

    if args.dir is not None:
        dir_ = args.dir
    
    if args.ablation is not None:
        ablation = args.ablation

    if args.command == 'model_based':
        app.run(train_model_based, ['_'])
    elif args.command == 'model_free':
        app.run(train_model_free, ['_'])
    elif args.command == 'explain':
        if args.dir is None:
            raise ValueError("missing argument: '--dir'")
        app.run(explain, ['_'])
    else:
        raise NotImplementedError(f"Unkown command: {args.command}")
