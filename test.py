import torch
from core import Buffer, CausalMdp, scm, TaskInfo
import learning.causal_discovery as causal_discovery

import learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.set_printoptions(precision=4)

job_salaries = [5, 10, 20, 30]

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
        payment_o = scm.EndoVar((work, job), lambda w, j: w * job_salaries[j],
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
        money = np.random.normal(scale=20)
        happy = np.random.normal(scale=5)
        payment = np.random.uniform(high=5)
        job = np.random.randint(len(job_salaries))
        work = np.random.rand()
        play = np.random.rand()

        return (money, happy, payment, job), (work, play)


mdp = MyMdp()
mdp.plot().view('./causal_graph.gv')


task = TaskInfo()
task.state('money', 'money(new)')
task.state('happy', 'happy(new)')
task.state('payment', 'payment(new)')
task.state('job', 'job(new)', categorical=True, dtype=np.uint8, 
           shape=len(job_salaries))
task.outcome('reward')
task.action('work', (), False, float)
task.action('play', (), False, float)

'''
parent_dic = {
    var.name: [pa.name for pa in var.parents]
    for var in mdp.endo_variables
}
'''

def sample_func():
    s, a = mdp.sample()
    mdp.model(s, a)
    return mdp.valuedic()


trainer = learning.Train(learning.Train.Config(
    learning.CausalNet.Config(task, 'cuda'),
    batchsize=128, niter_epoch=100, abort_window=10,
    batchsize_eval=256, num_sampling=500, conf_decay=0.1,
    causal_prior=0.4, causal_pvalue_thres=0.05, buffersize=5000,
), sampler = sample_func)


trainer.collect_samples(1000)
info = trainer.run(10, 'verbose')
info.show()


state, action = mdp.sample()
ae = learning.ActionEffect(
    trainer.network,
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
