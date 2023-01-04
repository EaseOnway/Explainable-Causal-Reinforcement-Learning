import os
from typing import Any, Optional, Final, final, List, Type, Dict
import numpy as np
import random
import torch
import json
from pathlib import Path
import abc

import argparse

import absl.app as app
from learning.base import Context, RLBase
from learning.config import Config
from ._env_setting import get_env_class, get_default_config


class Args:
    
    def __init__(self, **kargs):
        for k, v in kargs.items():
            setattr(self, k, v)

    def __str__(self):
        return "{" + \
            ', '.join([f"{k} = {v}" for k, v in vars(self)]) + \
        "}"

    def __repr__(self):
        return str(self)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(vars(self), f)
    
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            dic = json.load(f)
        return Args(**dic)
    
    def __getattr__(self, __name: str) -> Any:
        raise AttributeError


_REGISTERED: Dict[str, Type['Experiment']] = {}


class Experiment(abc.ABC, RLBase):
    
    ROOT: Final = Path('./experiments/')

    use_existing_path: bool

    @final
    @staticmethod
    def register(command: str, cls: Type['Experiment']):
        _REGISTERED[command] = cls
    
    @final
    @staticmethod
    def run(argv: Optional[List[str]] = None):
        if argv is None:
            import sys
            argv = sys.argv[1:]
        
        try:
            command = argv[0]
            argv = argv[1:]
        except ValueError:
            raise ValueError("usage: python <file> <command> ...")
        
        try:
            command = command.replace('-', '_')
            expcls = _REGISTERED[command]
        except KeyError:
            print(f"unkonwn command: {command}")
            print("supported commands:")
            for key in _REGISTERED.keys():
                doc = _REGISTERED[key].__doc__ or "no document"
                print(f"- {key}: {doc}")
            print("use '<command> --help' to see the usage of the command")
            raise ValueError("unsupported command.")
        
        exp = expcls(argv)
        app.run(exp.execute, ['_'])

    @classmethod
    def init_parser(cls, parser: argparse.ArgumentParser):
        if cls.use_existing_path:
            parser.add_argument('path', type=str, help='the experiment path')
            parser.add_argument('--seed', type=int, help='random seed')
        else:
            parser.add_argument('env', type=str, help='environment identifier')
            parser.add_argument('--config', '--cfg', type=str,
                                help='path of experiment configuration (json)')
            parser.add_argument('--run-id', type=str, help='path of the experiment data')
            parser.add_argument('--seed', type=int, help='random seed')

    @final
    def __init__(self, argv: List[str]):
        # get argument
        print("parsing arguments ....")
        parser = argparse.ArgumentParser(prog=self.__class__.__name__,
                                         description=self.__class__.__doc__)
        self.init_parser(parser)
        exp_args, argv = parser.parse_known_args(argv)
        
        if not self.use_existing_path:
            # get environment
            self.env_id: str = exp_args.env

            # setup environemnt
            print(f"setting up environment: {self.env_id}")
            env_class = get_env_class(self.env_id)
            parser = argparse.ArgumentParser(prog=self.__class__.__name__,
                                            description=self.__class__.__doc__)
            env_class.init_parser(parser)
            env_args, argv = parser.parse_known_args(argv)
            env = env_class(env_args)

            self.args = Args(**vars(exp_args), **vars(env_args))

            if len(argv) > 0:
                print("warning. unexpected arguments: " + (' '.join(argv)))

            # initializing
            print("setting up experiment")
            self.title = self.make_title()
            path, config = self.__setup()
            self.path = path
            RLBase.__init__(self, Context(config, env))
            self.setup()
            
            # save arguments
            self.args.save(self._file_path('args', 'json'))
            self.config.save(self._file_path('config', 'json'))
            print(f"successfully initialized experiment at {self.path}")
        
        else:
            self.args = Args(**vars(exp_args))
            if len(argv) > 0:
                print("warning. unexpected arguments: " + (' '.join(argv)))

            print("resolving configuration")
            old_args, config = self.__load_existing(exp_args)
            self.env_id = old_args.env

            print("creating environment:", self.env_id)
            env_class = get_env_class(self.env_id)
            env = env_class(old_args)

            print("setting up experiment")
            self.title = self.make_title()
            self.path = Path(self.args.path)
            RLBase.__init__(self, Context(config, env))
            self.setup()
            
            # save arguments
            self.args.save(self._file_path('args', 'json'))
            self.config.save(self._file_path('config', 'json'))
            print(f"successfully initialized experiment at {self.path}")


    @staticmethod
    def seed(x: int):
        torch.manual_seed(x)
        torch.cuda.manual_seed_all(x)
        np.random.seed(x)
        random.seed(x)

    @final
    def __load_existing(self, exp_args):
        assert self.use_existing_path
        path = Path(exp_args.path)

        if not path.exists():
            raise FileNotFoundError("path does not exist")
        args = Args.load(path / "args.json")

        config = Config()
        config.load(path / "config.json")

        return args, config
        
    @final
    def __setup(self):
        args = self.args
        
        run_id: str = args.run_id
        parent = Experiment.ROOT / self.env_id / self.title
        if not parent.exists():
            os.makedirs(parent)

        # seed
        seed = args.seed
        if seed is not None:
            Experiment.seed(seed)
        
        # get run_id
        if run_id is None or len(run_id) == 0:
            all_runs = os.listdir(parent)
            i = 0
            while True:
                i += 1
                run_id = "run-%d" % i
                if run_id not in all_runs:
                    break
        
        # experiment path
        path = parent / run_id
        if not path.exists():
            print("creating experiment directory at", path)
            os.makedirs(path)
        else:
            print(f"{path} already exists")
            while True:
                temp = input("Are you sure to proceed ? [y / n]: ")
                if temp == 'n':
                    raise FileExistsError("cannot initalize experiment directory")
                elif temp == 'y':
                    break
        # get config
        config = self.configure()

        return path, config

    def configure(self):
        args = self.args
        config = get_default_config(self.env_id, self.args)
        if args.config is not None:
            config_path = Path(args.config).with_suffix('.json')
            with config_path.open('r') as f:
                print(f"loading config from {config_path}")
                config.load_dict(json.load(f))
        return config

    @abc.abstractmethod
    def setup(self):
        pass

    def make_title(self):
        return self.__class__.__name__.lower()

    @final
    def _file_path(self, name: str, fmt: Optional[str] = None):
        if self.use_existing_path:
            path = self.path / (self.title + '-' + name)
        else:
            path = self.path / name
        if fmt is not None:
            path = path.with_suffix('.' + fmt)
        return path

    @final
    def execute(self, _ = []):
        self.env.launch()
        print(f"successfully launched environment: {self.env_id}")
        print(f"Experiment {self.title} begins.")
        self.main()
        print(f"Experiment {self.title} finished")

    @abc.abstractmethod
    def main(self):
        raise NotImplementedError
