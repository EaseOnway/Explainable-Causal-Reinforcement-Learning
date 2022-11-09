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

        self.define('d', ['x', 'y'], lambda x, y: np.abs(x) + np.abs(y))
        self.define('g', ['x', 'y'],
                    lambda x, y: np.max(np.abs([x, y])) < 1.0)
    
    def init(self):
        x, y = np.random.uniform(1.0, 5.0, 2) * (1 - np.random.randint(0, 2, 2) * 2)
        return {'x': x, 'y': y}
    
    def done(self, transition) -> bool:
        d, g = u.Collections.select(transition, ['d', 'g'])
        if d > 10 or g:
            return True
        return False
    
    def reward(self, transition) -> float:
        d, g = u.Collections.select(transition, ['d', 'g'])
        if d > 10:
            return -10
        elif g == True:
            return 10
        else:
            return - d / 10

    def random_action(self):
        return {'vx': np.random.normal(), 'vy': np.random.normal()}
        # return {'vx': np.random.randint(3), 'vy': np.random.randint(3)}


mdp = MyMdp()
# mdp.scm.plot().view('./causal_graph.gv')

config = cfg.Config(mdp)
config.causal_args.buffersize = 20000
config.ppo_args.buffersize = 2000
config.rl_args.discount = 0.9
config.rl_args.model_ratio = 0.9
config.rl_args.max_model_tr_len = 8
config.causal_args.n_sample_warmup = 1000
config.causal_args.pthres_independent = 0.1
config.causal_args.pthres_likeliratio = 0.1
config.device = torch.device('cuda')
config.ppo_args.gae_lambda = 0.9
config.ppo_args.entropy_penalty = 0.01
config.causal_args.optim_args.lr = 0.001
config.ppo_args.kl_penalty = 0.1
config.causal_args.n_iter_train = 100
config.causal_args.n_iter_eval = 8
config.causal_args.optim_args.batchsize = 512
config.ppo_args.optim_args.batchsize = 512
config.ppo_args.n_epoch_actor = 5
config.ppo_args.n_epoch_critic = 20
config.ppo_args.optim_args.lr = 0.001
# config.ablations.graph_fixed = True


trainer = learning.Train(config, "test")

# trainer.causal_graph = mdp.scm.parentdic()


trainer.run(100, 'verbose')


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
