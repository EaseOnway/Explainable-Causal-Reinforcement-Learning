from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import torch
import numpy as np

from .train import Train

from learning.buffer import Buffer
from learning.planning import Actor

from utils import Log


_LL = 'loglikelihood'
_NLL_LOSS = 'NLL loss'


class Fitting(Train):
    use_existing_path = False

    @classmethod
    def init_parser(cls, parser):
        super().init_parser(parser)
        parser.add_argument('--actor', type=str, default=None,
            help="path of the saved actor network. by default, the actor will be randomly initialized.")
        parser.add_argument('--test-size', type=int, default=1000)
        parser.add_argument('--train-size', type=int, default=10000)
        parser.add_argument('--n-step', type=int, default=20)
        parser.add_argument('--n-batch', type=int, default=20000)
        parser.add_argument('--ablation', type=str, default=None)
        parser.add_argument('--explore', type=float, default=None,
            help="the probability to take random actions rather than following the actor")

    def make_title(self):
        title = "fitting"
        ablation = self.args.ablation
        if self.args.ablation is None:
            return title
        else:
            return title + '-' + ablation

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

    @property
    def env_model(self):
        return self.env_models[0]
    
    def save_all(self):
        for i, model in enumerate(self.env_models):
            self.save(model.state_dict(), f'env-model-{i}', 'nn')
        if not self.config.ablations.mlp:
            self.save(self.causal_graph, 'causal-graph', 'json')

    def main(self):
        writer = self.writer
        test_size: int = self.args.test_size
        train_size: int = self.args.train_size
        explore_rate = self.args.explore
        n_step: int = self.args.n_step
        n_batch: int = self.args.n_batch

        # collect train samples
        test = Buffer(self.context)

        print("collecting test samples")
        self.collect(test, test_size, explore_rate, False, self.actor)
        self.buffer_m.clear()
        
        interval = max(train_size // n_step, 1)
        for i in range(0, train_size, interval):
            print(f"test ({i + interval}/{train_size}):")
            print("  collecting samples")
            self.collect(self.buffer_m, interval, explore_rate, False, self.actor)

            # causal_reasoning
            self.causal_discovery()
            self.env_models.init_parameters()
            self.fit(n_batch, -1)

            # eval
            log = Log()
            self.fit_epoch(test, log, eval=True)

            # write summary
            writer.add_scalar('log-likelihood', -log[_NLL_LOSS].mean, len(self.buffer_m))
            writer.add_scalars('log_likelihood_variable',
                {k: log[_LL, k].mean for k in self.env.names_output}, len(self.buffer_m))

            # show info
            print(f"- total log-likelihood:\t{-log[_NLL_LOSS].mean}")
            for k in self.env.names_output:
                print(f"- log-likelihood of '{k}':\t{log[_LL, k].mean}")
        
            # save
            self.save_all()
