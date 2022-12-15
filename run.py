from typing import Dict, Callable, Any, List
import torch
import numpy as np
import learning
import learning.config as cfg
from learning import Explainner
from baselines import BaselineExplainner
from configs import make_config, N_WARM_UP
import sys
from absl import app
from argparse import ArgumentParser

np.set_printoptions(precision=4)


_commands: Dict[str, Callable[[List[str]], Callable]] = {}
_doc_commands: Dict[str, str] = {}


def command(func: Callable[[List[str]], Callable]):
    name = func.__name__
    _commands[name] = func

    doc = func.__doc__ or "no document"
    _doc_commands[name] = doc
    return func


@command
def train_explain(argv: List[str]):
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--dir', type=str, default=None, required=True)
    parser.add_argument('--n-batch', type=int, default=10000)
    parser.add_argument('--n-sample', type=int, default=20000)
    parser.add_argument('--baseline', action='store_true', default=False)
    args = parser.parse_args(argv)

    def main(_):
        config = make_config(args.env, model_based=True)
        trainer = learning.Train(config, "explain", 'verbose')
        trainer.init_run(args.dir)
        trainer.load_item('agent')
        
        from learning.buffer import Buffer

        if not args.baseline:
            trainer.collect(trainer.buffer_m, args.n_sample, None, False)
            trainer.causal_discovery()
            trainer.fit(args.n_batch)
            trainer.save_items('env-model')
        else:
            explainer = BaselineExplainner(trainer)
            explainer.train_saliency(args.n_sample, args.n_batch)
            explainer.train_q(args.n_sample, args.n_batch)
            explainer.save()
    return main


@command
def model_based(argv: List[str]):
    '''train RL agents using the world model.'''
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--ablation', type=str, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--n-step', type=int, default=300)
    parser.add_argument('--store-buffer', action='store_true', default=False)
    args = parser.parse_args(argv)

    if args.seed is not None:
      learning.Train.set_seed(args.seed)

    def main(_):
        config = make_config(args.env, True, args.ablation)
        
        ablation = args.ablation
        if ablation is not None:
            expname = 'model_based_' + ablation
        else:
            expname = 'model_based'
        
        trainer = learning.Train(config, expname, 'verbose')
        trainer.init_run(args.dir)
        n_warmup = args.warmup or N_WARM_UP[args.env]
        trainer.warmup(n_warmup, 1)
        trainer.iter_policy(args.n_step, model_based=True)

        if args.store_buffer:
            trainer.save_item('buffer')
    
    return main


@command
def model_free(argv: List[str]):
    print(argv)
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--n-step', type=int, default=300)
    parser.add_argument('--store-buffer', action='store_true', default=False)

    args = parser.parse_args(argv)

    if args.seed is not None:
      learning.Train.set_seed(args.seed)

    def main(_):
        config = make_config(args.env, False)
        trainer = learning.Train(config, "model_free", 'verbose')
        trainer.init_run(args.dir)
        trainer.iter_policy(args.n_step, model_based=False)

        if args.store_buffer:
            trainer.save_item('buffer')

    return main


@command
def reasoning(argv: List[str]):
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--buffer-path', type=str, default=None)
    parser.add_argument('--n-batch', type=int, default=1024*16)
    args = parser.parse_args(argv)

    def main(_):
        config = make_config(args.env, model_based=True)
        trainer = learning.Train(config, "reasoning", 'verbose')
        trainer.init_run(args.dir)
        trainer.load_item('buffer', args.buffer_path)
        trainer.causal_discovery()
        trainer.fit(args.n_batch)
        trainer.save_items('env-model')
    return main


@command
def fit_test(argv: List[str]):
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--agent-path', type=str, default=None)
    parser.add_argument('--test-size', type=int, default=1000)
    parser.add_argument('--train-size', type=int, default=10000)
    parser.add_argument('--n-step', type=int, default=20)
    parser.add_argument('--n-batch', type=int, default=20000)
    parser.add_argument('--ablation', type=str, default=None)
    parser.add_argument('--explore', type=float, default=None)
    args = parser.parse_args(argv)

    def main(_):
        from learning.buffer import Buffer
        config = make_config(args.env, True, args.ablation)

        ablation = args.ablation
        if ablation is not None:
            expname = 'fit_' + ablation
        else:
            expname = 'fit'

        trainer = learning.Train(config, expname, 'verbose')
        trainer.init_run(args.dir)
        trainer.load_item('agent', args.agent_path)
        trainer.iter_model(args.train_size, args.test_size, args.n_step,
                           args.n_batch, args.explore)

    return main


@command
def explain(argv):
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--thres', type=float, default=0.1)
    parser.add_argument('--len', type=int, default=5)
    parser.add_argument('--baseline', action='store_true', default=False)
    args = parser.parse_args(argv)

    def main(_):
        config = make_config(args.env, model_based=True)
        trainer = learning.Train(config, "explain", 'plot')
        trainer.init_run(args.dir)
        # trainer.plot_causal_graph().view()

        trainer.warmup(5, 0)
        tran = trainer.buffer_m.arrays[3]
        s = trainer.env.state_of(tran)
        a = trainer.env.action_of(tran)

        if not args.baseline:
            trainer.load_items('agent', 'env-model')
            e = Explainner(trainer)
            e.why(s, a,
                  mode=True, thres=args.thres, maxlen=args.len, complete=True)
            e.whynot(s, trainer.env.random_action(),
                     mode=True, thres=args.thres, maxlen=args.len)
        else:
            e = BaselineExplainner(trainer)
            e.load()
            e.why(s, a)
    
    return main


if __name__ == "__main__":
    argv = sys.argv
    try:
        cmd = argv[1]
        cmd = cmd.replace('-', '_')
        func = _commands[cmd]
    except IndexError:
        print("require positional argument <command>")
        print("usage: python run.py <command> *[command arguments]")
        exit()
    except KeyError:
        print(f"unkonwn command: {argv[1]}")
        print("supported commands:")
        for key in _commands.keys():
            print(f"- {key}: {_doc_commands[key]}")
        print("use '<command> --help' to see the usage of the command")
        exit()
    
    command_args = argv[2:]
    main = func(command_args)
    app.run(main, ['_'])
