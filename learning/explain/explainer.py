from typing import Optional, Sequence, List

import numpy as np
import torch

from ..train import Train
from utils.typings import NamedValues
from ..base import Configured


from .explanan import Explanan, TrajectoryGenerator

class Explainner(Configured):

    def __init__(self, trainer: Train):
        super().__init__(trainer.config)

        self.trainer = trainer
        self.causnet = trainer.causnet

    def summarize(self, explanans: Sequence[Explanan]):
        rewards = dict.fromkeys(self.env.rewarders.keys(), 0.)
        for e in explanans:
            for k in e.rewards:
                rewards[k] += e.rewards[k] * e.discount
        total_reward = sum(rewards.values())
        
        print(f"In summary, after {len(explanans)} steps, the action at step 0" +
              " would likely cause a return about %.5f, " % total_reward +
              "where")
        for k in rewards:
            if rewards[k] != 0:
                print(f"|\t{rewards[k]} results from {k} ")

        if explanans[-1].terminated:
            print(f"Then, the episode terminates.")
        else:
            e = explanans[-1]
            nodes = e.nodes & set(self.env.names_next_s)
            if len(nodes) > 0:
                print(f"Then, the state becomes")
                for s in nodes:
                    print(f"|\t{s} = {e.variables[s]} ")
 
    def why(self, state: NamedValues, action: NamedValues,
            maxlen: Optional[int] = None,
            thres=0.1, mode=False, complete=False):

        np.set_printoptions(precision=5)

        trgen = TrajectoryGenerator(self.trainer, thres, mode=mode)
        trgen.reset(state)
        trgen.intervene(action)
        
        explanans = []
            
        for i in range(maxlen or 99999):
            transition = trgen.step()

            if transition is None:
                break

            e = Explanan(self.env, transition, thres, complete)
            explanans.append(e)
            print(e)

            if len(e.next_state_nodes) == 0:
                break
        
        self.summarize(explanans)

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
        
        explanans_f: List[Explanan] = []
        explanans_cf: List[Explanan] = []
            
        for i in range(maxlen or 99999):
            transition_f = trgen_f.step()
            transition_cf = trgen_cf.step()

            flag1, flag2 = False, False

            if transition_f is not None:
                e_f = Explanan(self.env, transition_f, thres, complete)
                explanans_f.append(e_f)
                if len(e_f.next_state_nodes) == 0:
                    flag1 = True
            else:
                flag1 = True

            if transition_cf is not None:
                e_cf = Explanan(self.env, transition_cf, thres, complete)
                explanans_cf.append(e_cf)
                if len(e_cf.next_state_nodes) == 0:
                    flag2 = True
            else:
                flag2 = True

            if flag1 and flag2:
                break
        
        temp = min(len(explanans_cf), len(explanans_f))
        for i in range(temp):
            e_f, e_cf = explanans_f[i], explanans_cf[i]
            print(e_f.compare(e_cf, eps))
        
        if temp < len(explanans_cf):
            print("However, the counterfactual episode continues.")
            for i in range(temp, len(explanans_cf)):
                print(explanans_cf[i])
        
        if temp < len(explanans_f):
            print("However, the factual episode continues.")
            for i in range(temp, len(explanans_f)):
                print(explanans_f[i])

        print("[Factual] ", sep='')
        self.summarize(explanans_f)

        print("[Counterfactual] ", sep='')
        self.summarize(explanans_cf)
