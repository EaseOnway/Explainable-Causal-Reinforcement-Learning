from typing import Any, Dict, Optional, Sequence, Tuple, Union, Iterable, Set

import numpy as np
import torch

from .causal_model.inferrer import Inferrer
from .causal_model import CausalNet, SimulatedEnv
from .train import Train
from .data import Batch
from utils.visualize import plot_digraph
from utils.typings import NamedTensors, NamedValues, Edge, SortedNames
from .config import Config
from .base import Configured

import core.scm as scm
from core import Env


_ACTION_EFFECT = "action_effect"
_CAUSAL_WEIGHT = "causal_weight"


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
    def __init__(self, trainer: Train, mode=False):
        self.env = SimulatedEnv(trainer.causnet)
        self.mode = mode
        self.trainer = trainer
        
        self.__action: Optional[NamedValues]
        self.__causal_weight: Dict[str, float]
        self.__terminated: bool

    @property
    def causal_weight(self):
        return self.__causal_weight

    def reset(self, state: NamedValues, action: NamedValues):
        self.__terminated = False
        self.__action = action
        self.__causal_weight = {name: 1.0 for name in self.env.names_s}
        self.env.reset(state, mode=self.mode)
    
    def __compute_causal_weight(self, ae: ActionEffect):
        for name_out in self.env.names_outputs:
            w = 0.
            for name_in in ae.who_cause(name_out):
                w += self.__causal_weight[name_in] * ae[name_in, name_out]
            if self.__action is not None:
                w += ae[name_out]
            self.__causal_weight[name_out] = w
        
        new = {s: self.__causal_weight[self.env.name_next(s)]
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
            next_causal_weight = self.__compute_causal_weight(ae)
            transition.info[_ACTION_EFFECT] = ae
            transition.info[_CAUSAL_WEIGHT] = self.__causal_weight
            self.__causal_weight = next_causal_weight
            
            self.__action = None
            return transition
    
    @staticmethod
    def _average_causal_weight(names: SortedNames,
                               trgens: Iterable['TrajectoryGenerator']):
        n = 0
        dic = {name: 0. for name in names}
        for trgen in trgens:
            n += 1
            weights = trgen.causal_weight
            for name in names:
                dic[name] += weights[name]
        for name in names:
            dic[name] /= n
        return dic
            

class Explainner(Configured):

    class MinimalExplanan:
        def __init__(self, step: int, discount: float, env: Env,
                     transition: Env.Transition, thres=0.1):
            self.__ae: ActionEffect = transition.info[_ACTION_EFFECT]
            self.__weights: Dict[str, float] = transition.info[_CAUSAL_WEIGHT]

            self.outcome_names: SortedNames = tuple(
                name for name in env.names_o
                if self.__weights[name] >= thres)

            state_names: Set[str] = set()
            for o in self.outcome_names:
                for cau in self.__ae.who_cause(o):
                    if self.__weights[cau] >= thres:
                        state_names.add(cau)
            self.variables = transition.variables
            self.action = self.__ae.action
            self.state_names: SortedNames = tuple(sorted(state_names))
            self.reward = transition.reward
            self.discounted_reward = self.reward * (discount**step)
            self.terminated = transition.terminated
            self.step = step

            self.max_outcome_weight = max(self.__weights[name] for name
                                          in env.names_o)
            self.sum_state_weight = sum(self.__weights[name] for name
                                        in env.names_next_s)

        def __str__(self):
            lines = []
            lines.append(f"At step {self.step}:")
            
            if len(self.state_names) > 0:
                if self.step == 0:
                    lines.append("|\tthe states are:")
                    for s in self.state_names:
                        lines.append(f"|\t|\t{s} = {self.variables[s]}")
                else:
                    lines.append("|\tdue to the former decisions,")
                    for s in self.state_names:
                        lines.append(f"|\t|\tstate {s} may possibly be {self.variables[s]} "
                                    "(weighted by %.4f);" % self.__weights[s])

            if self.step == 1:
                lines.append(f"|\twe take the given action:")
            else:
                lines.append(f"|\tProbably, we would take an action like:")
            for k, v in self.action.items():
                lines.append(f"|\t|\t{k} = {v}")
            
            if len(self.outcome_names) > 0:
                lines.append(f"|\tthen causing the outcomes")
                for o in self.outcome_names:
                    lines.append(f"|\t|\t{o} = {self.variables[o]} "
                                 "(weighted by %.4f);" % self.__weights[o])
                
                lines.append("|\tTherefore,")
                lines.append(f"|\t|\tthe reward is {self.reward}; and")
                lines.append(f"|\t|\tthe discounted reward is {self.discounted_reward}.")
                if self.terminated:
                    lines.append(f"|\tMeanwhile, the episode terminates.")

            return '\n'.join(lines)

    def __init__(self, trainer: Train):
        super().__init__(trainer.config)

        self.trainer = trainer
        self.causnet = trainer.causnet

    def explain(self, state: NamedValues, action: NamedValues,
                maxlen: Optional[int] = None, thres=0.1, mode=False):
        trgen = TrajectoryGenerator(self.trainer, mode=mode)
        trgen.reset(state, action)
        
        discount = self.config.rl_args.discount

        for i in range(maxlen or 99999):
            transition = trgen.step()
            if transition is None:
                break
            
            e = Explainner.MinimalExplanan(i, discount, self.env,
                                           transition, thres)
            
            if e.max_outcome_weight < thres:
                continue
            
            if e.sum_state_weight < thres:
                break

            print(e)
