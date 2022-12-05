from typing import Any, Dict, Optional, Sequence, Tuple, Union, Iterable, Set

import numpy as np
import torch

from ..causal_model import SimulatedEnv, CausalNetEnsemble
from ..train import Train
from utils.typings import NamedValues
from core import Env

from .action_effect import ActionEffect


_ACTION_EFFECT = "__action_effect__"
_CAUSAL_NODES = "__causal_nodes__"
_STEP = "__step__"
_DISCOUNT = "__discount__"


class TrajectoryGenerator:
    def __init__(self, trainer: Train, thres: float, mode=False):
        self.env: SimulatedEnv
        self.mode = mode
        self.trainer = trainer
        self.thres = thres
        self.discount = trainer.config.rl_args.discount

        if isinstance(self.trainer.causnet, CausalNetEnsemble):
            self.causnet = self.trainer.causnet.get_random_net()
        else:
            self.causnet = self.trainer.causnet
        
        self.__action: Optional[NamedValues]
        self.__terminated: bool
        self.__step: int
        self.__discount: float

    def reset(self, state: NamedValues):
        self.env = SimulatedEnv(self.causnet, state, self.mode)
        self.env.reset()

        self.__terminated = False
        self.__causal_nodes: Set[str] = set(self.env.names_s)
        self.__action = None
        self.__step = 0
        self.__discount = 1.0

    def intervene(self, action: NamedValues):
        self.__action = action

    def __update_causal_nodes(self, ae: ActionEffect):
        for name_out in self.env.names_outputs:
            flag = False
            
            if len(self.env.names_s) == 0 or self.__step == 0:
                if ae[name_out] >= self.thres:
                    flag = True
            else:
                causal_states = self.__causal_nodes & set(self.env.names_s)
                if len(causal_states) > 0 and ae[name_out] >= self.thres:
                    flag = True
            
            for name_in in ae.who_cause(name_out):
                if flag:
                    break
                if name_in in self.__causal_nodes and ae[name_in, name_out] >= self.thres:
                    flag = True

            if flag:
                self.__causal_nodes.add(name_out)
         
        new = {s for s, s_ in self.env.nametuples_s if s_ in self.__causal_nodes}
        return new

    def step(self):
        if self.__terminated:
            return None
        else:
            a = self.trainer.ppo.act(self.env.current_state, mode=self.mode)
            partial = None
            if self.__action is not None:
                partial = set(self.__action.keys())
                a.update(self.__action)
                self.__action = None 
            transition = self.env.step(a)
            self.__terminated = transition.terminated
            
            ae = ActionEffect(self.causnet, a, partial)
            
            next_causal_nodes = self.__update_causal_nodes(ae)
            transition.info[_ACTION_EFFECT] = ae
            transition.info[_CAUSAL_NODES] = self.__causal_nodes
            self.__causal_nodes = next_causal_nodes
            transition.info[_DISCOUNT] = self.__discount
            transition.info[_STEP] = self.__step
            self.__step += 1
            self.__discount *= self.discount
            return transition


class Explanan:
    def __init__(self, env: Env, transition: Env.Transition,
                 thres: float, complete=False):
        self.__env = env
        self.__ae: ActionEffect = transition.info[_ACTION_EFFECT]
        self.nodes = transition.info[_CAUSAL_NODES]

        self.variables = transition.variables
        self.vartexts = env.texts(transition.variables)
        self.action = env.texts(self.__ae.action)
        self.discount: float = transition.info[_DISCOUNT]
        self.total_reward = transition.reward
        self.terminated = transition.terminated
        self.step: int = transition.info[_STEP]

        self.next_state_nodes = self.nodes & set(env.names_next_s)

        self.rewards: Dict[str, float] = {}
        self.sources: Dict[str, Set[str]] = {}
        for label, rewarder in env.rewarders.items():
            sources = set(rewarder.source) & self.nodes
            if len(sources) == 0:
                continue
            self.rewards[label] = rewarder(self.variables)
            self.sources[label] = sources

        if complete:
            self.intervals = self.__get_complete_intervals(env)
        else:
            self.intervals = self.__get_minimal_intervals()
        self.interval_states = self.intervals & set(env.names_s)
        self.interval_outcomes = self.intervals & set(env.names_o)
        self.interval_nextstates = self.intervals & set(env.names_next_s)

    def __get_minimal_intervals(self):
        nodes: Set[str] = set()
        for source in self.sources.values():
            nodes = nodes | source
        return nodes

    def __get_complete_intervals(self, env: Env):
        minimal_nodes = self.__get_minimal_intervals()
        supplements: Set[str] = set()
        for node in env.names_outputs:
            if node in minimal_nodes:
                for cau in self.__ae.who_cause(node):
                    if cau in self.nodes:
                        supplements.add(cau)
            if node in self.nodes:
                supplements.add(node)
        return minimal_nodes | supplements

    def __str__(self):
        lines = []
        lines.append(f"At step {self.step}:")

        if len(self.interval_states) > 0:
            if self.step == 0:
                lines.append("|\twe have states:")
                for s in self.interval_states:
                    lines.append(f"|\t|\t{s} = {self.vartexts[s]}")
            else:
                lines.append("|\tdue to the former decisions, states will possibly be")
                for s in self.interval_states:
                    lines.append(f"|\t|\t{s} = {self.vartexts[s]} ")

        if self.step == 0:
            lines.append(f"|\tWe take the given action:")
        else:
            lines.append(f"|\tWe will possibly take an action like:")
        for k, v in self.action.items():
            lines.append(f"|\t|\t{k} = {v}")

        if len(self.interval_outcomes) > 0:
            lines.append(f"|\tThese will cause the following outcomes:")
            for o in self.interval_outcomes:
                lines.append(f"|\t|\t{o} = {self.vartexts[o]} ") 

        if len(self.interval_nextstates) > 0:
            if len(self.interval_outcomes) > 0:
                lines.append(f"|\tand states will possibly transit to")
            else:
                lines.append(f"|\tThese will cause the states to possibly transit to")
            for s in self.interval_nextstates:
                lines.append(f"|\t|\t{s} = {self.vartexts[s]} ")

        if len(self.intervals) > 0:
            lines.append("|\tTherefore, we obtained rewards (discounted by %.4f)" % self.discount)
            for label, reward in self.rewards.items():
                lines.append(f"|\t|\t{reward * self.discount} due to {label}")

        if self.terminated:
            lines.append(f"|\tFinally, The episode terminates.")

        return '\n'.join(lines)
    
    def __eq(self, name: str, x1, x2, eps=1e-3):
        v = self.__env.var(name)
        t1 = v.raw2input(v.tensor(x1, 'cpu').unsqueeze_(0))
        t2 = v.raw2input(v.tensor(x2, 'cpu').unsqueeze_(0))
        if torch.norm(t2 - t1) <= eps:
            return True
        else:
            return False

    def compare(self, other: 'Explanan', eps=1e-3):
        if self.step != other.step:
            raise ValueError("cannot compare explannans at different steps")

        def list_variables(names: Set[str], names_other: Set[str],
                           n_ind = 2, dif = True):
            names_x_only = names - names_other
            names_y_only = names_other - names
            names_both = names & names_other

            ind = n_ind * '|\t'
            lines = []

            for name in names_x_only:
                x = self.vartexts[name]
                lines.append(ind + f"{name} = {x} (factual only)")
            
            for name in names_both:
                x = self.vartexts[name]
                y = other.vartexts[name]
                if not self.__eq(name, self.variables[name],
                                 other.variables[name], eps):
                    lines.append(ind + f"{name} = {x} (factual) other than " +
                                 f"{y} (counterfactual)")
                elif dif is False:
                    lines.append(ind + f"{name} = {x}")

            for name in names_y_only:
                y = other.vartexts[name]
                lines.append(ind + f"{name} = {y} (counterfactual only)")

            return lines
            
        lines = []
        lines.append(f"At step {self.step}:")

        temp = list_variables(self.interval_states, other.interval_states)
        if len(temp) > 0:
            if self.step == 0:
                lines.append("|\twe have states:")
                lines.extend(temp)
            else:
                lines.append("|\tdue to the former decisions, states will possibly be")
                lines.extend(temp)

        action_keys = set(self.action.keys())
        temp = list_variables(action_keys, action_keys, dif=False)
        if self.step == 0:
            lines.append(f"|\tWe take the given action:")
        else:
            lines.append(f"|\tWe will possibly take an action like:")
        lines.extend(temp)

        temp = list_variables(self.interval_outcomes, other.interval_outcomes)
        flag = len(temp) > 0
        if flag:
            lines.append(f"|\tThese will cause the following outcomes:")
            lines.extend(temp)
        
        temp = list_variables(self.interval_nextstates,
                              other.interval_nextstates)
        if len(temp) > 0:
            if flag:
                lines.append(f"|\tand states will possibly transit to")
            else:
                lines.append(f"|\tThese will cause the states to possibly transit to")
            lines.extend(temp)

        if len(self.intervals) > 0:
            lines.append("|\tTherefore, we obtained rewards (discounted by %.4f)" % self.discount)
            for label, reward in self.rewards.items():
                reward_ = other.rewards[label]
                if abs(reward_ - reward) > eps:
                    lines.append(f"|\t|\t{reward * self.discount} (factual) due to {label}, "
                                 f"other than {reward_ * other.discount} (counterfactual)")

        if self.terminated:
            lines.append(f"|\tFinally, The  episode terminates.")

        return '\n'.join(lines)
