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
from core import Env, Distributions


_ACTION_EFFECT = "__action_effect__"
_CAUSAL_WEIGHTS = "__causal_weights__"
_STEP = "__step__"
_DISCOUNT = "__discount__"


class ActionEffect(Configured):
    def __init__(self, network: CausalNet, action: NamedValues,
                 intervened_names: Optional[Set[str]] = None):

        super().__init__(network.config)

        self.network = network
        self.action = action

        if intervened_names is None:
            self.intervened = None
        else:
            self.intervened = {name: action[name] for name in intervened_names}

        for a in self.env.names_a:
            if a not in action:
                raise ValueError(f"missing action variable {a}")

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

                if self.intervened is not None and \
                        len(set(parents_a) & self.intervened.keys()) == 0:
                    causal_weights[var] = {i: 0. for i in parents_s}
                    causal_weight_action[var] = 0.
                else:
                    causal_weights[var] = {i: float(w) for i, w
                                           in zip(parents_s, weight_s)}
                    causal_weight_action[var] = float(weight_a)

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
        self.discount = trainer.config.rl_args.discount
        
        self.__action: Optional[NamedValues]
        self.__causal_weights: Dict[str, float]
        self.__terminated: bool
        self.__step: int
        self.__discount: float

    def reset(self, state: NamedValues):
        self.env = SimulatedEnv(self.trainer.causnet, state, self.mode)
        self.env.reset()

        self.__terminated = False
        self.__causal_weights = {name: 0. for name in self.env.names_s}
        self.__action = None
        self.__step = 0
        self.__discount = 1.0

    def intervene(self, action: NamedValues):
        self.__action = action

    def __update_causal_weights(self, ae: ActionEffect):
        for name_out in self.env.names_outputs:
            w = 0.
            if len(self.env.names_s) == 0 or self.__step == 0:
                w += ae[name_out]
            else:
                w += np.mean([self.__causal_weights[k]
                             for k in self.env.names_s]) * ae[name_out]
            for name_in in ae.who_cause(name_out):
                pre_w = 1. if self.__step == 0 else self.__causal_weights[name_in]
                w += pre_w * ae[name_in, name_out]
            
            if w < self.thres:
                w = 0.

            self.__causal_weights[name_out] = w
         
        new = {s: self.__causal_weights[self.env.name_next(s)]
               for s in self.env.names_s}
        return new

    def step(self):
        if self.__terminated:
            return None
        else:
            a = self.trainer.ppo.act(self.env.current_state)
            partial = None
            if self.__action is not None:
                partial = set(self.__action.keys())
                a.update(self.__action)
                self.__action = None 
            transition = self.env.step(a)
            self.__terminated = transition.terminated
            ae = ActionEffect(self.trainer.causnet, a, partial)
            next_causal_weights = self.__update_causal_weights(ae)
            transition.info[_ACTION_EFFECT] = ae
            transition.info[_CAUSAL_WEIGHTS] = self.__causal_weights
            self.__causal_weights = next_causal_weights
            transition.info[_DISCOUNT] = self.__discount
            transition.info[_STEP] = self.__step
            self.__step += 1
            self.__discount *= self.discount
            return transition


class Explainner(Configured):

    class Explanan:
        def __init__(self, env: Env, transition: Env.Transition,
                     thres: float, complete=False):
            self.__ae: ActionEffect = transition.info[_ACTION_EFFECT]
            self.weights: Dict[str, float] = transition.info[_CAUSAL_WEIGHTS]
            self.nodes = set(name for name, weight in self.weights.items()
                               if weight >= thres)

            self.variables = transition.variables
            self.action = self.__ae.action
            self.discount: float = transition.info[_DISCOUNT]
            self.total_reward = transition.reward
            self.terminated = transition.terminated
            self.step: int = transition.info[_STEP]

            self.max_state_weights: float = max(
                self.weights[k] for k in env.names_next_s)

            self.reward_weights: Dict[str, float] = {}
            self.rewards: Dict[str, float] = {}
            self.sources: Dict[str, Set[str]] = {}
            for label, rewarder in env.rewarders.items():
                sources = set(rewarder.source) & self.nodes
                if len(sources) == 0:
                    continue
                w =  np.mean(
                    [self.weights[node] for node in sources])
                if w >= thres:
                    self.rewards[label] = rewarder(self.variables)
                    self.sources[label] = sources
                    self.reward_weights[label] = w

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
                        lines.append(f"|\t|\t{s} = {self.variables[s]}")
                else:
                    lines.append("|\tdue to the former decisions,")
                    for s in self.interval_states:
                        lines.append(f"|\t|\tstate {s} will possibly be {self.variables[s]} "
                                     "(causal weight = %.4f)" % self.weights[s])

            if self.step == 0:
                lines.append(f"|\tWe take the given action:")
            else:
                lines.append(f"|\tWe will possibly take an action like:")
            for k, v in self.action.items():
                lines.append(f"|\t|\t{k} = {v}")

            if len(self.interval_outcomes) > 0:
                lines.append(f"|\tThese will cause the following outcomes:")
                for o in self.interval_outcomes:
                    lines.append(f"|\t|\t{o} = {self.variables[o]} "
                                 "(causal weight = %.4f)" % self.weights[o]) 

            if len(self.interval_nextstates) > 0:
                if len(self.interval_outcomes) > 0:
                    lines.append(f"|\tand states will possibly transit to")
                else:
                    lines.append(f"|\tThese will cause the states to possibly transit to")
                for s in self.interval_nextstates:
                    lines.append(f"|\t|\t{s} = {self.variables[s]} "
                                 "(causal weight = %.4f)" % self.weights[s])

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

    def summarize(self, explanans: Sequence[Explanan]):
        rewards = dict.fromkeys(self.env.rewarders.keys(), 0.)
        reward_weights = dict.fromkeys(self.env.rewarders.keys(), 0.)
        for e in explanans:
            for k in e.rewards:
                rewards[k] += e.rewards[k]
                reward_weights[k] += e.reward_weights[k] / len(explanans)
        total_reward = sum(rewards.values())
        
        print(f"In Summary, after {len(explanans)} steps, the action at step 0" +
              " would likely cause a return about %.5f, " % total_reward +
              "where")
        for k in rewards:
            print(f"|\t{rewards[k]} results from {k} "
                  f"(average causal weight = {reward_weights[k]})")

        if explanans[-1].terminated:
            print(f"Then, the episode terminates.")
        else:
            e = explanans[-1]
            nodes = e.nodes & set(self.env.names_next_s)
            if len(nodes) > 0:
                print(f"Then, the state becomes")
                for s in nodes:
                    print(f"|\t{s} = {e.variables[s]} "
                          f"(causal weight = {e.weights[s]})")
 
    def explain(self, state: NamedValues, why: NamedValues,
                why_not: Optional[NamedValues] = None,
                maxlen: Optional[int] = None,
                thres=0.1, mode=False, complete=False):

        np.set_printoptions(precision=5)

        trgen = TrajectoryGenerator(self.trainer, thres, mode=mode)
        trgen.reset(state)
        trgen.intervene(why)

        if why_not is not None:
            raise NotImplementedError

        explanans = []
        for i in range(maxlen or 99999):
            transition = trgen.step()
            if transition is None:
                break

            e = Explainner.Explanan(self.env, transition, thres, complete)
            explanans.append(e)
            print(e)

            if e.max_state_weights < thres:
                break
        
        self.summarize(explanans)
