from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .causal_model.inferrer import Inferrer
from .causal_model import CausalNet
from .train import Train
from .data import Batch
from utils.visualize import plot_digraph
from utils.typings import NamedTensors, NamedValues
from .config import Config
from .base import Configured

import core.scm as scm


class ActionEffect(Configured):
    def __init__(self, network: CausalNet, action: Dict[str, Any],
                 attn_thres=0.):
        super().__init__(network.config)
        self.network = network
        self.attn_thres = attn_thres

        # compute:
        # 1. the encodings of action variables
        # 2. the action-effect embedding of each structural function
        # 3. the attention scores of each structrual function

        raw = Batch.from_sample(self.as_raws(action))

        with torch.no_grad():
            batch = raw.kapply(self.raw2input)
            actions = network.encoder.forward_all(batch)
            a_embs: NamedTensors = {}
            causations: Dict[str, Tuple[str, ...]] = {}
            weights: NamedTensors = {}
            for var in self.env.names_outputs:
                parents_a = network.parent_dic_a[var]
                parents_s = network.parent_dic_s[var]
                actions_pa = self.T.safe_stack([actions[pa] for pa in parents_a],
                                               (1, self.config.dims.action_encoding))
                inferrer = self.network.inferrers[var]
                a_emb = inferrer.aggregator.forward(actions_pa)
                kstates = self.network.k_model.forward(parents_s)
                attn = inferrer.get_attn_scores(a_emb, kstates)

                causations[var] = parents_s
                weights[var] = attn
        
        self.actions = actions
        self.__embs_a = a_embs
        self.__causations = causations
        self.__weights = weights

        self.__main_causations = {
            name: tuple(cau for i, cau in enumerate(causations[name])
                        if weight[i] < attn_thres / len(causations[name]))
            for name, weight in weights.items()}
    
    def infer(self, var: str, causal_states: Sequence[Any]):
        with torch.no_grad():
            weight = self.__weights[var]
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
            temp: torch.Tensor = inferrer.attn_infer(weight, emb_a, states)
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
                for cau, w in zip(self.__causations[var], self.__weights[var]):
                    print(f"\tcaused by {cau} with weight {float(w)}")
            else:
                print("\tno causations")

    def plot(self, format='png', main=True):
        return plot_digraph(
            self.config.env.names_outputs, self.__causations,  # type: ignore
            format=format)


class Explainner(Configured):
    def __init__(self, trainer: Train):
        super().__init__(trainer.config)

        self.trainer = trainer
        self.causnet = trainer.causnet
        self._env = trainer.causnet.create_simulated_env(random=False)
    
    def perdict_trajectory(self, state: NamedValues,
                           action: Optional[NamedValues], length: int):
        transitions = []
        action_effects = []
        logps_a = []
        logps_o = []
        logps = []

        self._env.reset(state)

        for _ in range(length):
            if action is None:
                action, logp_a = self.trainer.ppo.act(
                    self._env.current_state, compute_logp=True)
            else:
                s = Batch.from_sample(self.as_raws(self._env.current_state))
                a = Batch.from_sample(self.as_raws(action)).kapply(self.raw2label)
                pi = self.trainer.ppo.actor.forward(s)
                logp_a = float(pi.logprob(a))
            
            transition, reward, terminated, logp_o = self._env.step(action)
            assert isinstance(logp_a, float)
            assert isinstance(logp_o, float)
            logp = logp_a + logp_o
            ae = ActionEffect(self.causnet, action, 0.8)

            transitions.append(transition)
            action_effects.append(ae)
            logps_a.append(logp_a)
            logps_o.append(logp_o)
            logps.append(logp)

            action = None
        
        return


    def explain(self, state_action: NamedValues, horizon: int):
        raise NotImplementedError
