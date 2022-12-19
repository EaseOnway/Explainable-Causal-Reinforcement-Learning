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
    use_existing_path = True

    @classmethod
    def init_parser(cls, parser):
        super().init_parser(parser)
        parser.add_argument('--data', type=str, default=None,
            help="path of the data samples collected by some RL algorithm. "
                 "if given, disable --actor and --n-sample")
        parser.add_argument('--n-batch', type=int, default=10000)
        parser.add_argument('--n-sample', type=int, default=20000)
        parser.add_argument('--explainer', type=str, default='all',
            help='all, baseline, or causal')

    def make_title(self):
        return "explain"

    def setup(self):
        super().setup()
        self.env_models = self.creat_env_models(1)
        self.env_models.init_parameters()
        self.env_model_optimizers = self.env_models.optimizers()

        self.actor = Actor(self.context)
        self.actor.init_parameters()

        actor_path = self.path / "actor.nn"
        if actor_path.exists():
            self.actor.load(actor_path)
            print(f"successfully loaded actor from {actor_path}")
        else:
            print(f"Warning: {actor_path} does not exist. We will use untrained actor!")

        self.buffer_m = Buffer(self.context, self.config.model.buffer_size)
        if self.args.data is not None:
            self.buffer_m.load(self.args.data)
            print(f"successfully loaded data from {self.args.data}")
        
        if self.args.explainer not in ('all', 'causal', 'baseline'):
            raise ValueError(f"unknown explainer: {self.args.explainer}")
        self.train_causal = self.args.explainer in ('all', 'causal')
        self.train_baseline = self.args.explainer in ('all', 'baseline')

    def save_causal(self):
        model = self.env_models[0]
        self.save(model.state_dict(), f'env-model', 'nn')
        if not self.config.ablations.mlp:
            self.save(self.causal_graph, 'causal-graph', 'json')

    def main(self):
        if self.args.data is None:
            print("collecting data")
            self.collect(self.buffer_m, self.args.n_sample, None, False, self.actor)
        
        if self.train_causal:
            self.causal_discovery()
            self.fit(self.args.n_batch)
            self.save_causal()

        if self.train_baseline:
            baseline = BaselineExplainner(self.actor)
            baseline.train_q(self.buffer_m, self.args.n_batch)

            self.buffer_m.clear()
            print("collecting data for training saliency")
            self.collect(self.buffer_m, self.args.n_sample, 0, False, self.actor)
            baseline.train_saliency(self.buffer_m, self.args.n_batch)

            baseline.save(self._file_path("baseline", "nn"))
