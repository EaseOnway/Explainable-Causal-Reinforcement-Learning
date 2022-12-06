from typing import Optional, Sequence, List

import numpy as np
import torch

from ..train import Train
from utils.typings import NamedValues
from ..base import Configured


from .explanan import TrajectoryGenerator

class Explainner(Configured):

    def __init__(self, trainer: Train):
        super().__init__(trainer.config)

        self.trainer = trainer
        self.causnet = trainer.causnet
 
    def why(self, state: NamedValues, action: NamedValues,
            maxlen: Optional[int] = None,
            thres=0.1, mode=False, complete=False):

        np.set_printoptions(precision=5)

        trgen = TrajectoryGenerator(self.trainer, thres, mode=mode)
        trgen.reset(state)
        trgen.intervene(action)
            
        for i in range(maxlen or 99999):
            transition = trgen.step()
            if transition is None:
                break
            if len(trgen.causal_chain.next_vnodes) == 0:
                break
        
        for t in range(len(trgen.causal_chain)):
            print(trgen.causal_chain.explain(t, complete))
        trgen.causal_chain.summarize()
        

    def whynot(self, state: NamedValues, action_cf: NamedValues,
               action_f: Optional[NamedValues] = None,
               maxlen: Optional[int] = None, thres=0.1, mode=False, 
               complete=False, eps=1e-3):

        np.set_printoptions(precision=5)

        trgen_f = TrajectoryGenerator(self.trainer, thres, mode=mode)
        trgen_cf = TrajectoryGenerator(self.trainer, thres, mode=mode)
        trgen_f.reset(state)
        trgen_cf.reset(state)

        if action_f is None:
            action_f = self.trainer.ppo.act(state, mode=True)
        trgen_f.intervene({k: action_f[k] for k in action_cf})
        trgen_cf.intervene(action_cf)

        for i in range(maxlen or 99999):
            transition = trgen_f.step()
            if transition is None:
                break
            if len(trgen_f.causal_chain.next_vnodes) == 0:
                break
        
        for i in range(maxlen or 99999):
            transition = trgen_cf.step()
            if transition is None:
                break
            if len(trgen_cf.causal_chain.next_vnodes) == 0:
                break
        
        chain_f = trgen_f.causal_chain
        chain_cf = trgen_cf.causal_chain

        for t in range(max(len(chain_f), len(chain_cf))):
            print(chain_f.compare(t, chain_cf, eps, complete))

        print("[Factual] ", sep='')
        chain_f.summarize()

        print("[Counterfactual] ", sep='')
        chain_cf.summarize()
