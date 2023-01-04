from typing import Any, Dict, Optional, Sequence, Tuple, Union, Iterable, Set, List

import numpy as np
import torch

from ..env_model import CausalEnvModel
from ..planning import Actor
from utils.typings import NamedValues, SortedNames
from core import Env

from .action_effect import ActionEffect
import utils


_ACTION_EFFECT = "__action_effect__"

np.set_printoptions(precision=5)


class CausalNode:
    def __init__(self, name: str, value: Any, t: int, is_reward: bool, thres: float):
        self.name = name
        self.value = value
        self.t = t
        self.parents: Dict[CausalNode, Optional[float]] = {}
        self.is_reward = is_reward
        self.reached = False
        self.in_chain = False
        self.is_tail = False
        self.thres = thres

    def __back(self):
        assert self.in_chain
        for pa in self.parents:
            if not pa.in_chain:
                pa.in_chain = True
                pa.__back()
    
    def add_parent(self, node: "CausalNode", weight: Optional[float]):
        self.parents[node] = weight
        if node.reached and (weight is None or weight >= self.thres):
            self.reached = True
    
    def has_salient_parent(self, parent: "CausalNode"):
        try:
            w = self.parents[parent]
            return w is None or w >= self.thres
        except KeyError:
            return False

    def propagate(self):
        assert self.reached
        self.in_chain = True
        for parent in self.parents:
            if parent.reached and self.has_salient_parent(parent):
                parent.propagate()
            if self.is_reward:
                parent.is_tail = True

    def __str__(self):
        return f"{self.name} = {self.value} [t={self.t}, in_chain={self.in_chain}]"
    
    def __repr__(self):
        return str(self)


class CausalChain:
    def __init__(self, net: CausalEnvModel, thres: float,
                 from_: Optional[Set[str]] = None,
                 to: Optional[Set[str]] = None,
                 mode=True):
        self._variable_nodes: List[Dict[str, CausalNode]] = []
        self._reward_nodes: List[Dict[str, CausalNode]] = []
        self._transitions: List[Env.Transition] = []
        self._actions: List[NamedValues] = []
        self._thres = thres
        self._net = net
        self._mode = mode
        self._env = net.env
        self._discount = net.config.rl.discount
        self.__from = from_ or set(self._env.names_a)
        self.__to = to or set(self._env.rewarders.keys())

        self.ablation = net.ablations

    @property
    def next_vnodes(self):
        return self._variable_nodes[-1]
    
    @property
    def next_rnodes(self):
        return self._reward_nodes[-1]
    
    @property
    def t(self):
        return len(self._transitions)
    
    @property
    def terminated(self):
        return self.t > 0 and self._transitions[-1].terminated
    
    def __len__(self):
        return len(self._transitions)
    
    def step(self, transition: Env.Transition):
        assert not self.terminated
        if self.t == 0:
            ae = ActionEffect(self._net, self._env.action_of(transition.variables),
                              self.__from)
            self._variable_nodes.append({})
            self._reward_nodes.append({})
            for name in self._env.names_s:
                value = transition.variables[name]
                node = CausalNode(name, value, 0, False, 0)
                node.reached = True
                self.next_vnodes[name] = node
        else:
            ae = ActionEffect(self._net, self._env.action_of(transition.variables))

        for name_out in self._env.names_o:
            node = CausalNode(name_out, transition.variables[name_out], self.t, False, self._thres)
            for name_in in ae.who_cause(name_out):
                node.add_parent(self.next_vnodes[name_in],
                                None if self.ablation.no_attn else ae[name_in, name_out])
            if self.t == 0 and ae[name_out] >= self._thres: 
                node.reached = True
            self.next_vnodes[name_out] = node

        next_dict: Dict[str, CausalNode] = {}

        for name_s, name_out in self._env.nametuples_s:
            node = CausalNode(name_s, transition.variables[name_out], self.t + 1, False, self._thres)
            for name_in in ae.who_cause(name_out):
                node.add_parent(self.next_vnodes[name_in],
                                None if self.ablation.no_attn else ae[name_in, name_out])
            if self.t == 0 and ae[name_out] >= self._thres: 
                node.reached = True
            next_dict[name_s] = node
            self.next_vnodes[name_out] = node

        for label, rewarder in self._env.rewarders.items():
            r = rewarder(transition.variables)
            node = CausalNode(label, r, self.t, True, 0)
            for source in rewarder.source:
                node.add_parent(self.next_vnodes[source], None)
            self.next_rnodes[label] = node
            if node.name in self.__to and node.reached:
                node.propagate()
        
        self._actions.append(ae.action)
        self._transitions.append(transition)
        self._variable_nodes.append(next_dict)
        self._reward_nodes.append(dict())
    
    def __simulate(self, s: NamedValues, a: Union[NamedValues, Actor]):
        variables = s.copy()
        if isinstance(a, Actor):
            a = a.act(variables, self._mode)
        variables.update(a)
        outs = self._net.simulate(variables, self._mode)
        variables.update(outs)
        terminated = self._env.terminated(variables)
        reward = self._env.reward(variables)
        return Env.Transition(variables, reward, terminated)

    def follow(self, a: Union[NamedValues, Actor]):
        assert self.t > 0
        s = self._env.state_shift(self._transitions[-1].variables)
        transition = self.__simulate(s, a)
        self.step(transition)
    
    def start(self, s: NamedValues, a: Union[NamedValues, Actor]):
        transition = self.__simulate(s, a)
        self.step(transition)

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
                lines.append(f"|\t|\t{Env.name_next(s.name)} = {self.__text(s)}")

        rewards = [n for n in self._reward_nodes[t].values() if n.in_chain]
        discount = self._discount**t
        if len(rewards) > 0:
            lines.append("|\tTherefore, we obtained rewards (discounted by %.4f)" % discount)
            for node in rewards:
                lines.append("|\t|\t%.5f due to %s" % (node.value * discount, node.name))

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
            if not node.in_chain:
                return None
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
                    if abs(x.value - y.value) > eps:
                        lines.append(ind + 
                            "%.5f (factual) other than " % (x.value * discount) +
                            "%.5f (counterfactural) due to %s" % (y.value * discount, name))

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

    def plot(self, filename = 'causal-chain', only_chain=True):
        from graphviz import Digraph

        g = Digraph(filename, format='png')
        sub_gs = [Digraph(f"step_{t}")
                 for t in range(len(self) + 1)]
        
        def weight_color(w: float, is_chain_edge: bool):
            low = 0.2

            w = np.clip(w, 0., 1.)
            h = 0.33
            if is_chain_edge:
                s = low + (1 - low) * w
                v = 1.0
            else:
                s = 0.
                v = 1 - (low + (1 - low) * w)
            return "%.3f %.3f %.3f" % (h, s, v)

        def node_id(node: CausalNode):
            if node.is_reward:
                return f"reward({node.name})[{node.t}]"
            else:
                return f"{node.name}[{node.t}]"
        
        def node_label(node: CausalNode):
            if node.is_reward:
                return f"{node.name}[{node.t}]\n{round(node.value, 5)}"
            else:
                return f"{node.name}[{node.t}]\n" + \
                    f"{self._env.var(node.name).text(node.value)}"

        def set_node(node: CausalNode):
            
            if only_chain and not node.in_chain:
                return
            
            attr = {}
            if node.in_chain:
                if node.is_reward or node.is_tail:
                    attr["color"] = 'gold'
                else:
                    attr['color'] = 'green'

            if node.is_reward:
                attr['shape'] = 'hexagon'
            else:
                attr['shape'] = 'box'
            
            sub_gs[node.t].node(node_id(node), node_label(node), **attr)
            
            for parent, weight in node.parents.items():
                if only_chain and not parent.in_chain:
                    continue
                attr = {}
                if node.is_reward and parent.in_chain:
                    attr['color'] = 'gold'
                else:
                    attr['color'] = weight_color(
                        weight or 1.0, 
                        node.in_chain and parent.in_chain and
                        node.has_salient_parent(parent))
                if weight is not None:
                    attr['headlabel'] = "%.2f" % weight
                    attr['labeldistance'] = "1.5" 
                    attr['labelfontsize'] = "10.0"
                    attr['labelfontcolor'] = "blue" 
                    # attr['weight'] = str(weight)
                if not node.has_salient_parent(parent):
                    attr['style'] = 'dotted'
                
                if node.is_reward:
                    attr['dir'] = 'none'

                g.edge(node_id(parent), node_id(node), **attr)
        
        # nodes
        for t in range(len(self)):
            nodes = self._variable_nodes[t]
            if t == 0:
                for name in self._env.names_s:
                    set_node(nodes[name])
            for name in self._env.names_output:
                set_node(nodes[name])

            for r in self._reward_nodes[t].values():
                set_node(r)

        for t, sub_g in enumerate(sub_gs):
            g.subgraph(sub_g)

        return g
