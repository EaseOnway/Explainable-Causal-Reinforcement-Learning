import torch
from core import CausalMdp, scm, EnvInfo, ContinuousNormal, Categorical, Boolean
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import pandas as pd
import learning.config as cfg
import utils as u


np.set_printoptions(precision=4)


envinfo= EnvInfo()
envinfo.state('x', ContinuousNormal(scale=None))
envinfo.state('y', ContinuousNormal(scale=None))
envinfo.outcome('d', ContinuousNormal(scale=None))
envinfo.outcome('g', Boolean())
envinfo.action('vx', ContinuousNormal(scale=None))
envinfo.action('vy', ContinuousNormal(scale=None))



class MyMdp(CausalMdp):
    

    def __init__(self, dt=0.1):
        super().__init__(envinfo)
        
        self.define('x', ['x', 'vx'], lambda x, v: x + v * dt)
        self.define('y', ['y', 'vy'], lambda y, v: y + v * dt)

        vs = [-1, 0, 1]

        self.define('d', ['x', 'y'], lambda x, y: np.sqrt(x*x + y*y))
        self.define('g', ['x', 'y'],
                    lambda x, y: np.max(np.abs([x, y])) < 0.2)
    
    def init(self):
        return {'x': np.random.normal(), 'y': np.random.normal()}
    
    def done(self, transition, info) -> bool:
        x, y, g = u.Collections.select(transition, ['x\'', 'y\'', 'g'])
        if np.abs(x) > 5 or np.abs(y) > 5 or g == True:
            return True
        return False
    
    def reward(self, transition) -> float:
        g, d = u.Collections.select(transition, ['g', 'd'])
        return 10 if g else -d
    
    def random_action(self):
        return {'vx': np.random.normal(), 'vy': np.random.normal()}
        # return {'vx': np.random.randint(3), 'vy': np.random.randint(3)}


mdp = MyMdp()
# mdp.scm.plot().view('./causal_graph.gv')

config = cfg.Config(mdp)
config.causal_args.buffersize = 20000
config.ppo_args.buffersize = 2000
config.causal_args.n_sample_warmup = 2000
config.device = torch.device('cuda')
config.ppo_args.gamma = 0.9
config.ablations.graph_fixed = True
config.causal_args.n_iter_train = 50
config.causal_args.n_iter_eval = 5
config.causal_args.optim_args.batchsize = 512
config.ppo_args.optim_args.batchsize = 512
config.ppo_args.n_epoch_actor = 10
config.ppo_args.n_epoch_critic = 30
trainer = learning.Train(config, "test")
trainer.causal_graph = mdp.scm.parentdic()


trainer.run(500, 'verbose')


state = mdp.init()
action = mdp.random_action()

ae = learning.ActionEffect(
    trainer.causnet, action, attn_thres=0.2,
)


ae.print_info()
m = ae.create_causal_model()
m.plot().view('./action_effect_graph.gv')
m.assign(**state)
truth, r, done, info = mdp.step(u.Collections.merge_dic(state, action))
print(action)
for var in m.endo_variables:
    print(f"{var.name}: predicted[{var.value}], truth[{truth[var.name]}]")
