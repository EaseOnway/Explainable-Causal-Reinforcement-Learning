from typing import Optional, Callable
import torch

from envs import LunarLander
from core import Env
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import learning.config as cfg
import utils as u
from utils.typings import NamedValues
from learning import Explainner


def demo(trainer: learning.Train, expthres=0.1, explen=20, random=False):
    env = trainer.env
    env.reset()
    exp = Explainner(trainer)

    episode = 0
    i = 0
    while True:
        length = input()
        explain = False

        if length == 'q':
            break
        elif length == 'r':
            episode += 1
            i = 0
            env.reset()
        elif length == 'e':
            explain = True

        try:
            length = int(length)
        except ValueError:
            length = 1

        for _ in range(length):
            if random:
                a = env.random_action()
            else:
                a = trainer.ppo.act(env.current_state)
            
            transition = env.step(a)
            variables = transition.variables

            print(f"episode {episode}, step {i}:")
            print(f"| state:")
            for name in env.names_s:
                print(f"| | {name} = {variables[name]}")
            print(f"| action:")
            for name in env.names_a:
                print(f"| | {name} = {variables[name]}")
            print(f"| next state:")
            for name in env.names_next_s:
                print(f"| | {name} = {variables[name]}")
            print(f"| outcome:")
            for name in env.names_o:
                print(f"| | {name} = {variables[name]}")
            print(f"| reward = {transition.reward}")
            if len(transition.info) > 0:
                print(f"| info:")
                for k, v in transition.info.items():
                    (f"| | {k} = {v}")
            
            if explain:
                exp.why(env.state_of(transition.variables),
                        action=env.action_of(transition.variables),
                        maxlen=explen, thres=expthres, mode=True)

            if transition.terminated:
                print(f"episode {episode} terminated.")
                episode += 1
                i = 0
            else:
                i += 1

    env.reset()


# demo
if True:
    np.set_printoptions(precision=4)
    demo_env = LunarLander(render=True)
    config = cfg.Config()
    config.env = demo_env
    trainer = learning.Train(config, "demo")
    run_dir = "exemplar"
    trainer = learning.Train(config, "test", 'plot')
    trainer.init_run(f"experiments/LunarLander/test/{run_dir}",
                     resume=True)
    demo(trainer, expthres=0.15, explen=100)
