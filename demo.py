import torch

from envs import LunarLander
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import learning.config as cfg
import utils as u


np.set_printoptions(precision=4)

run = input()

demo_env = LunarLander(render=True)


config = cfg.Config(demo_env)

# config.ablations.graph_fixed = True

trainer = learning.Train(config, "demo")

# demo
trainer.load("experiments\\LunarLander\\test\\run-%s\\saved_state_dict" % run)
demo_env.demo(lambda s: trainer.ppo.act(s)[0])
