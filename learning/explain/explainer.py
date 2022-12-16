from typing import Optional, List, Set, Union, Tuple

import numpy as np

from ..train import Train
from utils.typings import NamedValues
from ..base import RLBase

from core import Env, Transitions

from learning.env_model import EnvNetEnsemble, CausalNet


from .explanan import CausalChain

class Explainner(RLBase):

    def __init__(self, trainer: Train):
        super().__init__(trainer.context)

        self.trainer = trainer

        net = trainer.envnet
        if isinstance(net, CausalNet):
            self.causnet = net
        elif isinstance(net, EnvNetEnsemble):
            net = net.get_random_net()
            if not isinstance(net, CausalNet):
                raise TypeError
            self.causnet = net
        else:
            raise TypeError
        
        self.actor = trainer.ppo.actor
 
    def build_chain(self, 
            startup: Union[Transitions, NamedValues, Tuple[NamedValues, NamedValues]],
            maxlen: int, from_: Optional[Set[str]] = None, to: Optional[Set[str]] = None,
            thres=0.1, mode=False):

        ch = CausalChain(self.causnet, thres, from_, to, mode)
        if isinstance(startup, dict):
            ch.start(startup, self.actor)
        elif isinstance(startup, tuple):
            s, a = startup
            ch.start(s, a)
        else:
            for i in range(min(maxlen, startup.n)):
                transition = startup.at(i)
                ch.step(transition)

        while ch.t < (maxlen or 9999):
            ch.follow(self.actor)
            if ch.terminated:
                break
        
        return ch

    def why(self, trajectory: Transitions, from_: Optional[Set[str]] = None,
            to: Optional[Set[str]] = None, maxlen: Optional[int] = None,
            thres=0.1, complete=True, mode=False):
        
        np.set_printoptions(precision=5)
        maxlen = maxlen or trajectory.n
        chain = self.build_chain(trajectory, maxlen, from_, to, thres, mode)

        for t in range(len(chain)):
            print(chain.explain(t, complete))
        chain.summarize()
        chain.plot(False).view()

    def whynot(self, trajectory: Transitions, action_cf: NamedValues,
               to: Optional[Set[str]] = None, maxlen: Optional[int] = None,
               thres=0.1, mode=False, complete=True, eps=1e-3):

        np.set_printoptions(precision=5)

        maxlen = maxlen or trajectory.n
        from_ = set(action_cf.keys())
        variables = trajectory.at(0).variables
        state = self.env.state_of(variables)
        action = self.env.action_of(variables)
        chain = self.build_chain((state, action), maxlen, from_, to, thres, mode)

        action.update(action_cf)
        chain_cf = self.build_chain((state, action), maxlen, from_, to, thres, mode)

        for t in range(max(len(chain), len(chain_cf))):
            print(chain.compare(t, chain_cf, eps, complete))

        print("[Factual] ", sep='')
        chain.summarize()

        print("[Counterfactual] ", sep='')
        chain_cf.summarize()
