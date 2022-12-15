from typing import Any, Dict, Optional, Sequence, Tuple, Union, Iterable, Set, List

import numpy as np
import torch

from ..env_model import SimulatedEnv, EnvNetEnsemble, CausalNet
from ..train import Train
from utils.typings import NamedValues, SortedNames
from core import Env

from .action_effect import ActionEffect


_ACTION_EFFECT = "__action_effect__"


class CausalNode:
    def __init__(self, name: str, value: Any, t: int, is_reward=False):
        self.name = name
        self.value = value
        self.t = t
        self.parents: List[CausalNode] = []
        self.is_reward = is_reward
        self.in_chain = is_reward
        self.is_tail = False

    def __back(self):
        assert self.in_chain
        for pa in self.parents:
            if not pa.in_chain:
                pa.in_chain = True
                pa.__back()
    
    def add_parent(self, node: "CausalNode"):
        self.parents.append(node)
        if self.in_chain:
            if not node.in_chain:
                node.in_chain = True
                node.__back()
        if self.is_reward:
            node.is_tail = True
    
    def __str__(self):
        return f"{self.name} = {self.value} [t={self.t}, in_chain={self.in_chain}]"
    
    def __repr__(self):
        return str(self)


class CausalChain:
    def __init__(self, env: Env, init_states: NamedValues, thres: float, discount: float):
        self._variable_nodes: List[Dict[str, CausalNode]] = [dict()]
        self._reward_nodes: List[Dict[str, CausalNode]] = [dict()]
        self._transitions: List[Env.Transition] = []
        self._actions: List[NamedValues] = []
        self._thres = thres
        self._env = env
        self._discount = discount
        for name, value in init_states.items():
            self.next_vnodes[name] = CausalNode(name, value, 0)
    
    @property
    def next_vnodes(self):
        return self._variable_nodes[-1]
    
    @property
    def next_rnodes(self):
        return self._reward_nodes[-1]
    
    @property
    def t(self):
        return len(self._transitions)
    
    def __len__(self):
        return len(self._transitions)
    
    def step(self, transition: Env.Transition):
        ae: ActionEffect = transition.info[_ACTION_EFFECT]

        for name_out in self._env.names_o:
            node = CausalNode(name_out, transition.variables[name_out], self.t)
            flag = ae[name_out] >= self._thres
            for name_in in ae.who_cause(name_out):
                if name_in in self.next_vnodes and ae[name_in, name_out] >= self._thres:
                    node.add_parent(self.next_vnodes[name_in])
                    flag = True
            if flag:
                self.next_vnodes[name_out] = node

        next_dict: Dict[str, CausalNode] = {}

        for name_s, name_out in self._env.nametuples_s:
            node = CausalNode(name_s, transition.variables[name_out], self.t + 1)
            flag = ae[name_out] >= self._thres
            for name_in in self._env.names_s:
                if name_in in self.next_vnodes and ae[name_in, name_out] >= self._thres:
                    node.add_parent(self.next_vnodes[name_in])
                    flag = True
            if flag:
                self.next_vnodes[name_out] = node
                next_dict[name_s] = node

        for label, rewarder in self._env.rewarders.items():
            sources = set(rewarder.source) & self.next_vnodes.keys()
            if len(sources) == 0:
                continue
            r = rewarder(transition.variables)
            node = CausalNode(label, r, self.t, True)
            for source in sources:
                node.add_parent(self.next_vnodes[source])
            self.next_rnodes[label] = node
        
        self._actions.append(ae.action)
        self._transitions.append(transition)
        self._variable_nodes.append(next_dict)
        self._reward_nodes.append(dict())

    def __get_intervals(self, t: int, names: SortedNames, complete=True):
            out: List[CausalNode] = []
            for name in names:
                if name in self._variable_nodes[t]:
                    node = self._variable_nodes[t][name]
                    if (complete and node.in_chain) or node.is_tail:
                        out.append(node)
            return out
        
    def __text(self, node: CausalNode):
        v = self._env.var(node.name)
        return v.text(node.value)

    def explain(self, t: int, complete=True):
        lines = []
        lines.append(f"At step {t}:")

        interval_states = self.__get_intervals(t, self._env.names_s, complete)
        if len(interval_states) > 0:
            if t == 0:
                lines.append("|\twe have states:")
                for s in interval_states:
                    lines.append(f"|\t|\t{s.name} = {self.__text(s)}")
            else:
                lines.append("|\tdue to the former decisions, states will possibly be")
                for s in interval_states:
                    lines.append(f"|\t|\t{s.name} = {self.__text(s)}")

        if self.step == 0:
            lines.append(f"|\tWe take the given action:")
        else:
            lines.append(f"|\tWe will possibly take an action like:")
        for k, v in self._actions[t].items():
            lines.append(f"|\t|\t{k} = {self._env.var(k).text(v)}")

        interval_outcomes = self.__get_intervals(t, self._env.names_o, complete)
        if len(interval_outcomes) > 0:
            lines.append(f"|\tThese will cause the following outcomes:")
            for o in interval_outcomes:
                lines.append(f"|\t|\t{o.name} = {self.__text(o)} ") 

        interval_nextstates = self.__get_intervals(t, self._env.names_next_s, complete)
        if len(interval_nextstates) > 0:
            if len(interval_outcomes) > 0:
                lines.append(f"|\tand states will possibly transit to")
            else:
                lines.append(f"|\tThese will cause the states to possibly transit to")
            for s in interval_nextstates:
                lines.append(f"|\t|\t{s.name} = {self.__text(s)} ")

        rewards = self._reward_nodes[t]
        discount = self._discount**t
        if len(rewards) > 0:
            lines.append("|\tTherefore, we obtained rewards (discounted by %.4f)" % discount)
            for label, reward in rewards.items():
                lines.append(f"|\t|\t{reward.value * discount} due to {label}")

        if self._transitions[t].terminated:
            lines.append(f"|\tFinally, The episode terminates.")

        return '\n'.join(lines)
    
    def __eq(self, name: str, x1, x2, eps=1e-3):
        v = self._env.var(name)
        t1 = v.raw2input(v.tensor(x1, 'cpu').unsqueeze_(0))
        t2 = v.raw2input(v.tensor(x2, 'cpu').unsqueeze_(0))
        if torch.norm(t2 - t1) <= eps:
            return True
        else:
            return False

    def __get_vnode(self, t: int, name: str, complete: bool = True):
        try:
            node = self._variable_nodes[t][name]
        except IndexError:
            return None
        except KeyError:
            return None
        if (complete and node.in_chain) or node.is_tail:
            return node
        else:
            return None
    
    def __get_rnode(self, t: int, name: str):
        try:
            node = self._reward_nodes[t][name]
        except IndexError:
            return None
        except KeyError:
            return None
        return node

    def compare(self, t: int, other: 'CausalChain', eps=1e-3, complete=True):

        def list_variables(names: Iterable[str], n_ind = 2):
            ind = n_ind * '|\t'
            lines = []

            for name in names:
                x = self.__get_vnode(t, name, complete)
                y = other.__get_vnode(t, name, complete)

                if x is not None and y is None:
                    lines.append(ind + f"{name} = {self.__text(x)} (factual only)")
                elif y is not None and x is None:
                    lines.append(ind + f"{name} = {self.__text(y)} (counterfactual only)")
                elif x is not None and y is not None:
                    if not self.__eq(name, x.value, y.value, eps):
                        lines.append(ind + f"{name} = {self.__text(x)} (factual) other than " +
                                    f"{self.__text(y)} (counterfactual)")

            return lines
        
        def list_rewards(discount: float, n_ind = 2):
            ind = n_ind * '|\t'
            lines = []

            for name in self._env.rewarders.keys():
                x = self.__get_rnode(t, name)
                y = other.__get_rnode(t, name)

                if x is not None and y is None:
                    lines.append(ind + f"{x.value * discount} (factual only) due to {name}")
                elif y is not None and x is None:
                    lines.append(ind + f"{y.value * discount} (factual only) due to {name}")
                elif x is not None and y is not None:
                    if not abs(x.value - y.value) > eps:
                        lines.append(ind + f"{x.value * discount} (factual) other than " +
                                     f"{y.value * discount} (counterfactural) due to {name}")

            return lines

        lines = []
        lines.append(f"At step {t}:")

        # state
        temp = list_variables(self._env.names_s)
        if len(temp) > 0:
            if self.step == 0:
                lines.append("|\twe have states:")
                lines.extend(temp)
            else:
                lines.append("|\tdue to the former decisions, states will possibly be")
                lines.extend(temp)

        # action
        if self.t == 0:
            lines.append(f"|\tWe take the action:")
        else:
            lines.append(f"|\tWe will possibly take an action like:")
        for name in self._env.names_a:
            x = self._actions[t][name]
            y = other._actions[t][name]
            text_x = self._env.var(name).text(x)
            text_y = self._env.var(name).text(y)
            if not self.__eq(name, x, y, eps):
                lines.append(f"|\t|\t{name} = {text_x} (factual) other than " +
                            f"{text_y} (counterfactual)")
            else:
                lines.append( f"|\t|\t{name} = {text_y}")

        # outcome
        temp = list_variables(self._env.names_o)
        flag = len(temp) > 0
        if flag:
            lines.append(f"|\tThese will cause the following outcomes:")
            lines.extend(temp)
        
        # next_state
        temp = list_variables(self._env.names_next_s)
        if len(temp) > 0:
            if flag:
                lines.append(f"|\tand states will possibly transit to")
            else:
                lines.append(f"|\tThese will cause the states to possibly transit to")
            lines.extend(temp)

        discount = self._discount**t
        temp = list_rewards(discount)
        if len(temp) > 0:
            lines.append("|\tTherefore, we obtained rewards (discounted by %.4f)" % discount)
            lines.extend(temp)

        if t < len(self) and self._transitions[t].terminated:
            lines.append(f"|\tThe factural episode terminates.")
        if t < len(other) and other._transitions[t].terminated:
            lines.append(f"|\tThe counterfactural episode terminates.")

        return '\n'.join(lines)
    
    def summarize(self):
        rewards = dict.fromkeys(self._env.rewarders.keys(), 0.)
        for t in range(len(self)):
            discount = self._discount**t
            for label, node in self._reward_nodes[t].items():
                rewards[label] += node.value * discount
        total_reward = sum(rewards.values())
        
        print(f"In summary, after {len(self)} steps, the action at step 0" +
              " would likely cause a return about %.5f, " % total_reward +
              "where")
        for k in rewards:
            if rewards[k] != 0:
                print(f"|\t{rewards[k]} results from {k} ")

        if self._transitions[-1].terminated:
            print(f"Then, the episode terminates.")
        else:
            nodes: List[CausalNode] = []
            for name in self._env.names_next_s:
                if name in self.next_vnodes:
                    nodes.append(self.next_vnodes[name])
            if len(nodes) > 0:
                print(f"Then, the state becomes")
                for s in nodes:
                    print(f"|\t{s.name} = {self.__text(s)} ")

class TrajectoryGenerator:
    def __init__(self, trainer: Train, thres: float, mode=False):
        self.env: SimulatedEnv
        self.mode = mode
        self.trainer = trainer
        self.thres = thres
        self.discount = trainer.config.rl.discount

        if isinstance(self.trainer.envnet, EnvNetEnsemble):
            envnet = self.trainer.envnet.get_random_net()
        else:
            envnet = self.trainer.envnet
        if not isinstance(envnet, CausalNet):
            raise TypeError("Require causal enviornment model")
        self.envnet = envnet
        
        self.__action: Optional[NamedValues]
        self.__terminated: bool
        self.__step: int
        self.__discount: float
        self.__causal_chain: CausalChain

    def reset(self, state: NamedValues):
        self.env = SimulatedEnv(self.envnet, state, self.mode)
        self.env.reset()

        self.__terminated = False
        self.__action = None
        self.__step = 0
        self.__discount = 1.0
        self.__causal_chain = CausalChain(self.trainer.env, 
            state, self.thres, self.discount)

    def intervene(self, action: NamedValues):
        self.__action = action

    def step(self):
        if self.__terminated:
            return None
        else:
            a = self.trainer.ppo.actor.act(self.env.current_state, mode=self.mode)
            partial = None
            if self.__action is not None:
                partial = set(self.__action.keys())
                a.update(self.__action)
                self.__action = None 
            transition = self.env.step(a)
            self.__terminated = transition.terminated
            
            ae = ActionEffect(self.envnet, a, partial)
            transition.info[_ACTION_EFFECT] = ae
            self.__causal_chain.step(transition)
            self.__step += 1
            self.__discount *= self.discount
            return transition

    @property
    def causal_chain(self):
        return self.__causal_chain
