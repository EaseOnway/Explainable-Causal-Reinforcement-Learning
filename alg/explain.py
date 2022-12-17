from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import torch
import numpy as np

from .train import Train

from learning.buffer import Buffer
from learning.planning import Actor
from baselines import BaselineExplainner

from utils import Log


_LL = 'loglikelihood'
_NLL_LOSS = 'NLL loss'


class TrainExplain(Train):

    @classmethod
    def init_parser(cls, parser):
        super().init_parser(parser)
        parser.add_argument('--actor', type=str, default=None,
            help="path of the saved actor for collecting data.")
        parser.add_argument('--data', type=str, default=None,
            help="path of the data samples collected by some RL algorithm. "
                 "if given, disable --actor and --n-sample")
        parser.add_argument('--n-batch', type=int, default=10000)
        parser.add_argument('--n-sample', type=int, default=20000)

    def make_title(self):
        return "explain"

    def setup(self):
        super().setup()
        self.env_models = self.creat_env_models(1)
        self.env_models.init_parameters()
        self.env_model_optimizers = self.env_models.optimizers()

        self.actor = Actor(self.context)
        self.actor.init_parameters()
        if self.args.actor is not None:
            self.actor.load(self.args.actor)
            print(f"successfully loaded actor from {self.args.actor}")
        
        self.buffer_m = Buffer(self.context, self.config.model.buffer_size)
        if self.args.data is not None:
            self.buffer_m.load(self.args.data)
            print(f"successfully loaded data from {self.args.data}")
    
    def save_all(self):
        for i, model in enumerate(self.env_models):
            self.save(model.state_dict(), f'env-model-{i}', 'nn')
        if not self.config.ablations.mlp:
            self.save(self.causal_graph, 'causal-graph', 'json')

    def main(self):
        if self.args.data is None:
            print("collecting data")
            self.collect(self.buffer_m, self.args.n_sample, None, False, self.actor)
        
        self.causal_discovery()
        self.fit(self.args.n_batch)

        baseline = BaselineExplainner(self.actor)
        baseline.train_q(self.buffer_m, self.args.n_batch)

        self.buffer_m.clear()
        print("collecting data for training saliency")
        self.collect(self.buffer_m, self.args.n_sample, 0, False, self.actor)
        baseline.train_saliency(self.buffer_m, self.args.n_batch)

        self.save_all()
        baseline.save(self._file_path("baselines", "nn"))
        print("successfully saved baselines") 
