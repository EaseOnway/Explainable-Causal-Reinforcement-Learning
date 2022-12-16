import os
from typing import Any, Optional, Final
import numpy as np
import random
import torch
import json
import traceback
import abc

from ..learning.config import Config
from ..learning.base import RLBase, Context


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

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(vars(self), f)
    
    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            dic = json.load(f)
        return Args(**dic)
    
    def __getattr__(self, __name: str) -> Any:
        raise AttributeError



class Experiment:
    
    ROOT: Final[str] = './experiments/'

    def __init__(self, env_id: str, title: str, config: Config,
                 run_id: Optional[str], args: Args):

        self.env_id: Final = env_id
        self.title: Final = title
        print(f"Initializing experiment [{self.title}] at {self.path}")
        run_id, path = self.__get_path(run_id)
        self.run_id: Final = run_id
        self.path: Final = path
        self.args: Final = args
        context = Context(config)
        RLBase.__init__(self, context)
        print("Using following configuration:")
        print(config)

    @staticmethod
    def seed(x: int):
        torch.manual_seed(x)
        torch.cuda.manual_seed_all(x)
        np.random.seed(x)
        random.seed(x)

    def setup(self, env_id: str, title: str):
        d = Experiment.ROOT + self.env_id + '/' + self.title + '/'
        if not os.path.exists(d):
            print(f"creating experiment directory at {d}")
            os.makedirs(d)
        if not os.path.exists(d + 'config.json')
        

    def __get_path(self, run_id: Optional[str]):
        root = Experiment.ROOT + self.env_id + '/' + self.title + '/'
        
        if run_id is None:
            if not os.path.exists(root):
                os.makedirs(root)
            run_ids = os.listdir(root)
            i = 0
            while True:
                i += 1
                run_id = "run-%d" % i
                if run_id not in run_ids:
                    break
            path = root + run_id + '/'
            os.makedirs(path)
        else:
            if len(run_id) == 0:
                raise ValueError("run-id is empty")
            path = root + run_id + '/'
            if not os.path.exists(path):
                os.makedirs(path)
        return run_id, path

    def restore(self):
        raise NotImplementedError(f"experiment '{self.title}' cannot be restored")
    
    def save_runtime(self):
        pass

    def save(self):
        self.context.config.save(self.path + 'config.json')
        self.args.save(self.path + 'args.json')
        self.save_runtime()

    @classmethod
    def resume(cls, env_id: str, title: str, run_id: str):
        path = Experiment.ROOT + env_id + '/' + title + '/' + run_id + '/'
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        config = Config()
        config.load(path + 'config.json')
        args = Args.load(path + 'args.json')
        e = cls(env_id, title, config, run_id, args)
        e.restore()
        return e

    def main(self):
        try:
            print("executing experiment")
            self.save()
            self.execute()
            print("experiment finished successfully")
        except Exception as e:
            print("detected exception")
            traceback.print_exc()
            print("performing abort operation")
            self.abort()
            print("experiment finished unsuccessfully")

    def abort(self):
        self.save()
    
    @abc.abstractmethod
    def execute(self):
        raise NotImplementedError
