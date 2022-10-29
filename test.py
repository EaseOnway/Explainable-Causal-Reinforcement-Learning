import torch
from core import Buffer, CausalMdp, scm, EnvInfo
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import learning.config as cfg
import utils as u


np.set_printoptions(precision=4)


envinfo= EnvInfo()
envinfo.state('x')
envinfo.state('y')
envinfo.state('vx')
envinfo.state('vy')
envinfo.outcome('d')
envinfo.outcome('v')
envinfo.action('ax')
envinfo.action('ay')


class MyMdp(CausalMdp):

    def __init__(self, dt=0.1):
        super().__init__(envinfo, {'v': -1, 'd': -1})
        
        self.define('x', ['x', 'vx'], lambda x, v: x + v * dt)
        self.define('y', ['y', 'vy'], lambda y, v: y + v * dt)
        self.define('vx', ['vx', 'ax'], lambda v, a: v + a * dt)
        self.define('vy', ['vy', 'ay'], lambda v, a: v + a * dt)
        self.define('d', ['x', 'y'], lambda x, y: np.sqrt(x*x + y*y))
        self.define('v', ['vx', 'vy'], lambda x, y: np.sqrt(x*x + y*y))
    
    def init(self):
        return {'x': np.random.normal(), 'y': np.random.normal(), 'vx': 0, 'vy': 0}
    
    def done(self, transition, info) -> bool:
        x, y = u.basics.select(transition, ['x\'', 'y\''])
        if np.abs(x) > 5 or np.abs(y) > 5:
            return True
        return False
    
    def sample(self):
        return self._2karrays({'ax': np.random.normal(), 'ay': np.random.normal()})


mdp = MyMdp()
mdp.scm.plot().view('./causal_graph.gv')


config = cfg.Config(mdp,
    batchsize=128, n_iter_epoch=50, convergence_window=10,
    n_iter_planning = 20, check_convergence=True,
    batchsize_eval=256, n_sample_epoch=100,
    n_sample_warmup=1000, conf_decay=0.05, causal_prior=0.4,
    causal_pvalue_thres=0.05, buffersize=5000,
    device=torch.device('cuda'), gamma=0.98,
)

trainer = learning.Train("test", config)


trainer.collect_samples(1000)
trainer.run(50, 'verbose')


state = mdp.init()
action = mdp.sample()

ae = learning.ActionEffect(
    trainer.causnet,
    action,
    attn_thres=0.2,
)

ae.print_info()
m = ae.create_causal_model()
m.plot().view('./action_effect_graph.gv')
m.assign(**state)
truth, r, done, info = mdp.step(u.basics.merge_dic(state, action))
print(action)
for var in m.endo_variables:
    print(f"{var.name}: predicted[{var.value}], truth[{truth[var.name]}]")


