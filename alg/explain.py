from typing import Any, Callable, Dict, List, Literal, Optional, final, Tuple
import torch
import numpy as np
import json
from .train import Train
from .experiment import Experiment
from learning.explain import Explainner
from learning.explain.action_effect import ActionEffect
from learning.buffer import Buffer
from learning.planning import Actor
from baselines import BaselineExplainner
from learning.env_model import CausalEnvModel
from learning.buffer import Buffer
from core import Tag
from utils import Log
from utils.typings import ParentDict


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
        parser.add_argument('--explainer', type=str, default='causal',
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


class TestExplain(Experiment):
    use_existing_path = True

    def make_title(self):
        return 'test'

    def setup(self):
        super().setup()
        self.model = CausalEnvModel(self.context)
        self.actor = Actor(self.context)
        self.actor.load(self.path / 'actor.nn')
        self.model.load(self.path / 'explain-env-model.nn')
        self.explainer = Explainner(self.actor, self.model)
        # self.baseline = BaselineExplainner(self.actor)
        # self.baseline.load(self.path / 'explain-baseline.nn')

        with open(self.path / 'explain-causal-graph.json', 'r') as f:
            graph = json.load(f)
            self.model.load_graph(graph)

        self.trajectory = Buffer(self.context, 
             max_size=(self.config.rl.max_episode_length or 1000))
        self.episode = 0
        self.t = 0
    
    def __play(self, length: int):
        for _ in range(length):
            a = self.actor.act(self.env.current_state)
            transition = self.env.step(a)
            self.trajectory.write(transition.variables, transition.reward,
                                  Tag.encode(transition.terminated, False, False))
            variables = transition.variables

            print(f"episode {self.episode}, step {self.t}:")
            print(f"| state:")
            for name in self.env.names_s:
                print(f"| | {name} = {variables[name]}")
            print(f"| action:")
            for name in self.env.names_a:
                print(f"| | {name} = {variables[name]}")
            print(f"| next state:")
            for name in self.env.names_next_s:
                print(f"| | {name} = {variables[name]}")
            print(f"| outcome:")
            for name in self.env.names_o:
                print(f"| | {name} = {variables[name]}")
            print(f"| reward = {transition.reward}")
            if len(transition.info) > 0:
                print(f"| info:")
                for k, v in transition.info.items():
                    (f"| | {k} = {v}")
            
            if transition.terminated:
                print(f"episode {self.episode} terminated.")
                self.episode += 1
                self.t = 0
            else:
                self.t += 1

    def __loop(self):
        x = input("input a line ('h' for help): ")
        length = 0
        if x == 'h':
            print("- a integar to push forward k steps")
            print("- an empty line to push forward 1 step")
            print("- 'q' to quit.")
            print("- 'e' for an explanation.")
            print("- 'r' to reset environment")
        elif x == 'q':
            return -1
        elif x == 'r':
            self.env.reset()
        elif x == 'e':
            if len(self.trajectory) == 0:
                print("trajectory is empty!")
            else:
                self.explainer.why(
                    self.trajectory.transitions[-5:],
                    maxlen=5, thres=0.1, mode=True,
                    plotfile=str(self._file_path('causal-chain')),
                    # to={'getting close'} # define target rewards here
                )
                # self.explainer.whynot(
                #     self.trajectory.transitions[-5:], self.env.random_action(),
                #     maxlen=5, thres=0.1, mode=True,
                #     # to={'getting close'}
                # )
        elif x == '':
            length = 1
        else:
            try:
                length = int(x)
            except ValueError:
                print(f"Sorry! I don't know what {x} means...")

        self.__play(length)
        return 0

    def main(self):
        while(True):
            x = self.__loop()
            if x < 0:
                break
