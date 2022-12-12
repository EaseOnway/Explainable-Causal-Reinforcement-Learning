import torch
import numpy as np
from envs import SC2BuildMarine
import learning
import learning.config as cfg
from learning import Explainner
from configs import make_config, N_WARM_UP

np.set_printoptions(precision=4)


def train_temp(_):
    config = make_config(args.env, False)
    config.saliency.sparse_factor = 0.005
    trainer = learning.Train(config, "model_free", 'verbose')
    trainer.init_run(r"experiments\Cartpole\model_free\run-7", resume=True)
    # trainer.iter_policy(100, model_based=False)
    
    from learning.baselines.saliency.explainer import SaliencyExplainner
    exp = SaliencyExplainner(trainer)
    exp.train(10)
    
    trainer.warmup(5, 0.)
    tran = trainer.buffer_m.arrays[3]
    a = trainer.env.action_of(tran)
    exp.why(trainer.env.state_of(tran), a)

def train_model_based(_):
    config = make_config(args.env, True)
    
    ablation = args.ablation
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
    trainer.init_run(args.dir)
    trainer.warmup(N_WARM_UP[args.env], 1)
    trainer.iter_policy(300, model_based=True)


def train_model_free(_):
    config = make_config(args.env, False)
    trainer = learning.Train(config, "model_free", 'verbose')
    trainer.init_run(args.dir)
    trainer.iter_policy(300, model_based=False)
    trainer.causal_reasoning(1024 * 16)


def causal_resoning(_):
    config = make_config(args.env, model_based=True)
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run(args.dir, resume=True)
    trainer.warmup(N_WARM_UP[args.env], 1)
    trainer.warmup(N_WARM_UP[args.env], None)
    trainer.warmup(N_WARM_UP[args.env], 0)
    trainer.causal_reasoning(1024 * 16)
    trainer.save()


def explain(_):
    config = make_config(args.env, model_based=True)
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run(args.dir, resume=True)
    trainer.plot_causal_graph().view()
    exp = Explainner(trainer)
    trainer.warmup(5, 0)
    tran = trainer.buffer_m.arrays[3]
    a = trainer.env.action_of(tran)

    exp.why(trainer.env.state_of(tran), a,
            mode=True, thres=0.15, maxlen=10, complete=True)

    exp.whynot(trainer.env.state_of(tran), trainer.env.random_action(),
               mode=True, thres=0.15, maxlen=10)


if __name__ == "__main__":
    from absl import app
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('command', type=str, default='model_based',
                        help="'model_based', 'model_free', or 'explain'")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--ablation', type=str, default=None)
    args = parser.parse_args()

    if args.seed is not None:
      learning.Train.set_seed(args.seed)

    if args.command == 'model_based':
        app.run(train_model_based, ['_'])
    elif args.command == 'model_free':
        app.run(train_model_free, ['_'])
    elif args.command == 'explain':
        if args.dir is None:
            raise ValueError("missing argument: '--dir'")
        app.run(explain, ['_'])
    elif args.command == 'temp':
        app.run(train_temp, ['_'])
    else:
        raise NotImplementedError(f"Unkown command: {args.command}")
