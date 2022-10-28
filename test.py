import torch
from core import Buffer, CausalMdp, scm, EnvInfo
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import learning.config as cfg


np.set_printoptions(precision=4)

job_salaries = np.array([5, 10, 20, 30])


class MyMdp(CausalMdp):

    def __init__(self):
        super().__init__()
        
        money = scm.ExoVar(name='money')
        happy = scm.ExoVar(name='happy')
        payment = scm.ExoVar(name='payment')
        job = scm.ExoVar(name='job')

        work = scm.ExoVar(name='work')
        play = scm.ExoVar(name='play')

        money_o = scm.EndoVar((money, payment),
                              lambda m, p: m + p, name='money(new)')
        happy_o = scm.EndoVar((play,), lambda p: 5*p, name='happy(new)')
        payment_o = scm.EndoVar((work, job), lambda w, j: w * np.sum(job_salaries * j),
                                name='payment(new)')
        job_o = scm.EndoVar((job,), lambda a: a, name='job(new)')

        r = scm.EndoVar((money_o, happy_o), lambda m,
                        h: 0.5 * m + 0.5 * h, name='reward')

        self.config(
            state_vars=((money, money_o), (happy, happy_o),
                        (payment, payment_o), (job, job_o)),
            action_vars=(work, play),
            reward_vars=(r,)
        )

    def sample(self):
        money = np.random.uniform(0., 1.)
        happy = np.random.normal(scale=5)
        payment = np.random.uniform(high=5)
        
        job = np.random.randint(len(job_salaries))
        job = np.eye(len(job_salaries))[job]
        
        work = np.random.rand()
        play = np.random.rand()

        return (money, happy, payment, job), (work, play)






envinfo= EnvInfo()
envinfo.state('money')
envinfo.state('happy')
envinfo.state('payment')
envinfo.state('job', dtype=np.uint8, shape=len(job_salaries))
envinfo.outcome('reward')
envinfo.action('work', (), False, float)
envinfo.action('play', (), False, float)


class MyMdp(CausalMdp):

    def __init__(self):
        super().__init__(envinfo)
        
        self.define('money', ['money', 'payment'], lambda m, p: m + p)
        self.define('happy', ['play'], lambda p: 5*p)
        self.define('payment', ['work', 'job'])

        money_o = scm.EndoVar((money, payment),
                              lambda m, p: m + p, name='money(new)')
        happy_o = scm.EndoVar((play,), lambda p: 5*p, name='happy(new)')
        payment_o = scm.EndoVar((work, job), lambda w, j: w * np.sum(job_salaries * j),
                                name='payment(new)')
        job_o = scm.EndoVar((job,), lambda a: a, name='job(new)')

        r = scm.EndoVar((money_o, happy_o), lambda m,
                        h: 0.5 * m + 0.5 * h, name='reward')

        self.config(
            state_vars=((money, money_o), (happy, happy_o),
                        (payment, payment_o), (job, job_o)),
            action_vars=(work, play),
            reward_vars=(r,)
        )

mdp.plot().view('./causal_graph.gv')

'''
parent_dic = {
    var.name: [pa.name for pa in var.parents]
    for var in mdp.endo_variables
}
'''


config = cfg.Config(task,
    batchsize=128, niter_epoch=100, convergence_window=10,
    niter_planning_epoch = 50, check_convergence=True,
    batchsize_eval=256, num_sampling=500,
    conf_decay=0.05, causal_prior=0.4,
    causal_pvalue_thres=0.05, buffersize=5000,
    device=torch.device('cuda'), gamma=0.0,
)

trainer = learning.Train(config)


trainer.collect_samples(1000)
trainer.run(50, 'verbose')


state, action = mdp.sample()
ae = learning.ActionEffect(
    trainer.causnet,
    {"work": np.array(action[0]),
     "play": np.array(action[1])},
    attn_thres=0.2,
)

ae.print_info()
m = ae.create_causal_model()
m.plot().view('./action_effect_graph.gv')
m.assign(money=state[0],
         happy=state[1],
         payment=state[2],
         job=state[3])
mdp.model(state, action)
print("work = %f, play = %f" % action)
for var in m.endo_variables:
    print(f"{var.name}: predicted[{var.value}], truth[{mdp[var.name].value}]")


