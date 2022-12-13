from typing import Dict, Callable, Any, List
import torch
import numpy as np
from envs import SC2BuildMarine
import learning
import learning.config as cfg
from learning import Explainner
from configs import make_config, N_WARM_UP
import sys

np.set_printoptions(precision=4)


_commands: Dict[str, Callable[[List[str]], Callable]] = {}
_doc_commands: Dict[str, str] = {}


def command(func: Callable[[List[str]], Callable]):
    name = func.__name__
    _commands[name] = func

    doc = func.__doc__ or "no document"
    _doc_commands[name] = doc
    return func


'''
@command
def temp(argv: List[str]):
    config = make_config(args.env, False)
    config.saliency.sparse_factor = 0.005
    trainer = learning.Train(config, "model_free", 'verbose')
    trainer.init_run(r"experiments\Cartpole\model_free\run-7", resume=True)
    # trainer.iter_policy(100, model_based=False)
    
    from baselines.saliency.explainer import SaliencyExplainner
    exp = SaliencyExplainner(trainer)
    exp.train(10)
    
    trainer.warmup(5, 0.)
    tran = trainer.buffer_m.arrays[3]
    a = trainer.env.action_of(tran)
    exp.why(trainer.env.state_of(tran), a)
'''


@command
def model_based(argv: List[str]):
    '''train RL agents using the world model.'''
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--ablation', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=False)
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
        trainer.init_run(args.dir, resume=args.resume)
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
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--n-step', type=int, default=300)
    parser.add_argument('--store-buffer', action='store_true', default=False)

    args = parser.parse_args(argv)

    if args.seed is not None:
      learning.Train.set_seed(args.seed)

    def main(_):
        config = make_config(args.env, False)
        trainer = learning.Train(config, "model_free", 'verbose')
        trainer.init_run(args.dir, resume=args.resume)
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
        trainer.init_run(args.dir, resume=True)
        trainer.load_item('buffer', args.buffer_path)
        trainer.causal_discovery()
        trainer.fit(args.n_batch)
        trainer.save_items('env-model')
    return main


@command
def collect_samples(argv: List[str]):
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--n_train', type=int, default=1024, required=True)
    parser.add_argument('--n_test', type=int,  default=256, required=True)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--agent-path', type=str, default=None)
    
    args = parser.parse_args(argv)

    def main(_):
        from learning.buffer import Buffer
        config = make_config(args.env, model_based=True)
        buffer_train = Buffer(config, config.envmodel_args.buffer_size)
        buffer_test = Buffer(config, config.envmodel_args.buffer_size)

        trainer = learning.Train(config, "samples", 'verbose')
        trainer.init_run(args.dir, resume=True)

        e = 1.0 if args.agent_path is None else None
        if args.agent_path is not None:
            trainer.load_item('agent', args.agent_path)

        print("collecting training data...")
        trainer.collect(buffer_train, args.n_train, e)
        buffer_train.save(trainer.run_dir + 'buffer-train')
        
        print("collecting testing data...")
        trainer.collect(buffer_test, args.n_test, e)
        buffer_test.save(trainer.run_dir + 'buffer-test')

        print(f"done. data saved in {trainer.run_dir}")
    return main


@command
def fit_test(argv: List[str]):
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--data-dir', type=str, default=None, required=True)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--n-step', type=int, default=30)
    parser.add_argument('--n-batch', type=int, default=1024*16)
    parser.add_argument('--ablation', type=str, default=None)
    args = parser.parse_args(argv)

    def main(_):
        from learning.buffer import Buffer
        config = make_config(args.env, True, args.ablation)
        buffer_train = Buffer(config, config.envmodel_args.buffer_size)
        buffer_test = Buffer(config, config.envmodel_args.buffer_size)

        ablation = args.ablation
        if ablation is not None:
            expname = 'fit_' + ablation
        else:
            expname = 'fit'

        trainer = learning.Train(config, expname, 'verbose')
        trainer.init_run(args.dir)
        buffer_train.load(args.data_dir + '/buffer-train')
        buffer_test.load(args.data_dir + '/buffer-test')
        trainer.iter_model(buffer_train, buffer_test, args.n_step, args.n_batch)

    return main


@command
def explain(argv):
    parser = ArgumentParser()
    parser.add_argument('env', type=str, help="environment name")
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--thres', type=float, default=0.1)
    parser.add_argument('--len', type=int, default=5)
    args = parser.parse_args(argv)

    def main(_):
        config = make_config(args.env, model_based=True)
        trainer = learning.Train(config, "test", 'plot')
        trainer.init_run(args.dir, resume=True)
        trainer.plot_causal_graph().view()
        exp = Explainner(trainer)
        trainer.warmup(5, 0)
        tran = trainer.buffer_m.arrays[3]
        a = trainer.env.action_of(tran)

        exp.why(trainer.env.state_of(tran), a,
                mode=True, thres=args.thres, maxlen=args.len, complete=True)

        exp.whynot(trainer.env.state_of(tran), trainer.env.random_action(),
                mode=True, thres=args.thres, maxlen=args.len)

    return main


if __name__ == "__main__":
    from absl import app
    from argparse import ArgumentParser

    argv = sys.argv
    try:
        cmd = argv[1]
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
