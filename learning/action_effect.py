from typing import Any, Dict, Optional, Sequence, Tuple, Union, Iterable, Set

import numpy as np
import torch

from .causal_model.inferrer import Inferrer
from .causal_model import CausalNet, SimulatedEnv
from .train import Train
from core import Batch
from utils.visualize import plot_digraph
from utils.typings import NamedTensors, NamedValues, Edge, SortedNames
from .config import Config
from .base import Configured

import core.scm as scm
from core import Env


_ACTION_EFFECT = "action_effect"
_CAUSAL_WEIGHTS = "causal_weights"


class ActionEffect(Configured):
    def __init__(self, network: CausalNet, action: Dict[str, Any]):
        super().__init__(network.config)
        self.network = network

        # compute:
        # 1. the encodings of action variables
        # 2. the action-effect embedding of each structural function
        # 3. the attention scores of each structrual function

        raw = Batch.from_sample(self.as_raws(action))

        with torch.no_grad():
            batch = raw.kapply(self.raw2input)
            action_enc = network.encoder.forward_all(batch)
            a_embs: NamedTensors = {}
            causations: Dict[str, Tuple[str]] = {}
            attns: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            causal_weights: Dict[str, Dict[str, float]] = {}
            causal_weight_action: Dict[str, float] = {}

            for var in self.env.names_outputs:
                parents_a = network.parent_dic_a[var]
                parents_s = network.parent_dic_s[var]
                actions_pa = self.T.safe_stack([action_enc[pa] for pa in parents_a],
                                               (1, self.config.dims.action_encoding))
                inferrer = self.network.inferrers[var]
                a_emb = inferrer.aggregator.forward(actions_pa)
                kstates = self.network.k_model.forward(parents_s)
                attn = inferrer.get_attn_scores(a_emb, kstates)

                causations[var] = parents_s
                attns[var] = attn

                weight_s = self.T.t2a(attn[0].view(len(parents_s)))
                weight_a = float(attn[1].view(()))

                causal_weights[var] = {i: float(w) for i, w in zip(parents_s, weight_s)}
                causal_weight_action[var] = float(weight_a)
        
        self.action = action
        self.__action_enc = action_enc
        self.__embs_a = a_embs
        self.__causations = causations
        self.__attns = attns
        self.__causal_weights = causal_weights
        self.__causal_weight_action = causal_weight_action
        
    def __getitem__(self, key: Union[Edge, str]):
        if isinstance(key, str):
            return self.__causal_weight_action[key]
        else:
            name_in, name_out = key
            try:
                return self.__causal_weights[name_out][name_in]
            except KeyError:
                return 0.
    
    def who_cause(self, name: str):
        return self.__causations[name]
        
    def infer(self, var: str, causal_states: Sequence[Any]):
        with torch.no_grad():
            weight = self.__attns[var]
            emb_a = self.__embs_a[var]
            caus = self.__causations[var]
            states = []
            for pa, state in zip(caus, causal_states):
                raw = self.as_raw(pa, state)
                raw = raw.unsqueeze_(0)
                s = self.raw2input(pa, raw)
                states.append(self.network.encoder.forward(pa, s))
            states = self.T.safe_stack(states, (1, self.dims.state_encoding))
            inferrer = self.network.inferrers[var]
            temp: torch.Tensor = inferrer.attn_infer(*weight, emb_a, states)
            temp = inferrer.feed_forward(temp)
            distri = inferrer.decoder.forward(temp)
            temp = distri.mode
            temp = temp.squeeze(0)
            out = self.T.t2a(temp, self.v(var).dtype.numpy)
        return out
    
    def __get_causal_eq(self, var: str):
        def causal_eq(*args):
            return self.infer(var, args)
        return causal_eq

    def create_causal_model(self):
        m = scm.StructrualCausalModel()
        for var in self.env.names_s:
            m[var] = scm.ExoVar(name=var)
        for var in self.env.names_outputs:
            m[var] = scm.EndoVar(
                [m[pa] for pa in self.__causations[var]],
                self.__get_causal_eq(var),
                name = var,
            )
        return m

    def print_info(self):
        for var in self.env.names_outputs:
            print(f"exogenous variable '{var}':")
            if len(self.__causations[var]) > 0:
                for cau, w in zip(self.__causations[var], self.__attns[var]):
                    print(f"\tcaused by {cau} with weight {float(w)}")
            else:
                print("\tno causations")

    def plot(self, format='png', main=True):
        return plot_digraph(
            self.config.env.names_outputs, self.__causations,  # type: ignore
            format=format)


class TrajectoryGenerator:
    def __init__(self, trainer: Train, thres: float, mode=False):
        self.env: SimulatedEnv
        self.mode = mode
        self.trainer = trainer
        self.thres = thres
        
        self.__action: Optional[NamedValues]
        self.__causal_weights: Dict[str, float] = {}
        self.__terminated: bool

    def reset(self, variables: NamedValues):
        self.__terminated = False
        
        state = {k: variables[k] for k in self.trainer.env.names_s}
        action = {k: variables[k] for k in self.trainer.env.names_a}
        self.env = SimulatedEnv(self.trainer.causnet, state, self.mode)

        self.__action = action
        self.__causal_weights = {name: 1. for name in self.env.names_s}
        self.env.reset()
    
    def __update_causal_weights(self, ae: ActionEffect):
        for name_out in self.env.names_outputs:
            w = 0.
            if ae[name_out] >= self.thres:
                if len(self.env.names_s) == 0:
                    w += ae[name_out]
                else:
                    w += np.mean([self.__causal_weights[k]
                                  for k in self.env.names_s]) * ae[name_out]
            for name_in in ae.who_cause(name_out):
                if ae[name_in, name_out] >= self.thres:
                    w += self.__causal_weights[name_in] * ae[name_in, name_out]
            self.__causal_weights[name_out] = w
         
        new = {s: self.__causal_weights[self.env.name_next(s)]
               for s in self.env.names_s}
        return new

    def step(self):
        if self.__terminated:
            return None
        else:
            if self.__action is not None:
                a = self.__action
            else:
                a = self.trainer.ppo.act(self.env.current_state)
            
            transition = self.env.step(a)
            self.__terminated = transition.terminated
            ae = ActionEffect(self.trainer.causnet, a)
            next_causal_weights = self.__update_causal_weights(ae)
            transition.info[_ACTION_EFFECT] = ae
            transition.info[_CAUSAL_WEIGHTS] = self.__causal_weights
            self.__causal_weights = next_causal_weights
            
            self.__action = None
            return transition
            

class Explainner(Configured):

    class Explanan:
        def __init__(self, step: int, discount: float, env: Env,
                     transition: Env.Transition, thres: float,
                     complete=False):
            self.__ae: ActionEffect = transition.info[_ACTION_EFFECT]
            self.__weights: Dict[str, float] = transition.info[_CAUSAL_WEIGHTS]
            self.__nodes = set(name for name, weight in self.__weights.items()
                               if weight >= thres)

            self.variables = transition.variables
            self.action = self.__ae.action
            self.discount = (discount**step)
            self.total_reward = transition.reward
            self.terminated = transition.terminated
            self.step = step

            self.max_state_weights: float = max(
                self.__weights[k] for k in env.names_s)

            self.reward_weights: Dict[str, float] = {}
            self.rewards: Dict[str, float] = {}
            self.sources: Dict[str, Set[str]] = {}

            # self.sources: Dict[str, Set[str]] = {}
            for label, rewarder in env.rewarders.items():
                w = np.mean([self.__weights[node] for node in rewarder.source])
                if w >= thres:
                    self.rewards[label] = rewarder(self.variables)
                    self.reward_weights[label] = w
                    self.sources[label] = set(rewarder.source) & self.__nodes
            if complete:
                self.intervals = self.__get_complete_intervals(env)
            else:
                self.intervals = self.__get_minimal_intervals()
            self.interval_states = self.intervals & set(env.names_s)
            self.interval_outcomes = self.intervals & set(env.names_s)
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
                        if cau in self.__nodes:
                            supplements.add(cau)
                if node in self.__nodes:
                    supplements.add(node)
            return minimal_nodes | supplements
                            

        def __str__(self):
            lines = []
            lines.append(f"At step {self.step}:")
            
            if len(self.interval_states) > 0:
                if self.step == 0:
                    lines.append("|\twe have states:")
                    for s in self.interval_states:
                        lines.append(f"|\t|\t{s} = {self.variables[s]}")
                else:
                    lines.append("|\tdue to the former decisions,")
                    for s in self.interval_states:
                        lines.append(f"|\t|\tstate {s} will possibly be {self.variables[s]} "
                                     "(causal weight = %.4f)" % self.__weights[s])

            if self.step == 1:
                lines.append(f"|\tWe take the given action:")
            else:
                lines.append(f"|\tWe will possibly take an action like:")
            for k, v in self.action.items():
                lines.append(f"|\t|\t{k} = {v}")
            
            if len(self.interval_outcomes) > 0:
                lines.append(f"|\tThese will cause the following outcomes:")
                for o in self.interval_outcomes:
                    lines.append(f"|\t|\t{o} = {self.variables[o]} "
                                 "(causal weight = %.4f)" % self.__weights[o])
                
            
            if len(self.interval_nextstates) > 0:
                if len(self.interval_outcomes) > 0:
                    lines.append(f"|\tand states will possibly transit to")
                else:
                    lines.append(f"|\tThese will cause the states to possibly transit to")
                for s in self.interval_nextstates:
                    lines.append(f"|\t|\t{s} = {self.variables[s]} "
                                 "(causal weight = %.4f)" % self.__weights[s])
                    
            if len(self.intervals) > 0:
                lines.append("|\tTherefore, we obtained rewards (discounted by %.4f)" % self.discount)
                for label, reward in self.rewards.items():
                    lines.append(f"|\t|\t{reward * self.discount} due to {label}"
                                 "(causal weight = %.4f)" % self.reward_weights[label])
            
            if self.terminated:
                lines.append(f"|\tFinally, The episode terminates.")

            return '\n'.join(lines)

    def __init__(self, trainer: Train):
        super().__init__(trainer.config)

        self.trainer = trainer
        self.causnet = trainer.causnet

    def explain(self, state_action: NamedValues, maxlen: Optional[int] = None,
                thres=0.1, mode=False, complete=False):
        
        np.set_printoptions(precision=5)

        trgen = TrajectoryGenerator(self.trainer, thres, mode=mode)
        trgen.reset(state_action)
        
        discount = self.config.rl_args.discount

        for i in range(maxlen or 99999):
            transition = trgen.step()
            if transition is None:
                break
            
            e = Explainner.Explanan(i, discount, self.env, transition,
                                    thres, complete)
            
            if e.max_state_weights < thres:
                break
            
            print(e)
