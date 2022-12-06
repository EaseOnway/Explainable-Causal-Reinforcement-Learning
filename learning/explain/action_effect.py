from typing import Any, Dict, Optional, Sequence, Tuple, Union, Iterable, Set

import numpy as np
import torch

from ..causal_model.inferrer import Inferrer
from ..causal_model import CausalNet, SimulatedEnv
from ..train import Train
from core import Batch
from utils.visualize import plot_digraph
from utils.typings import NamedTensors, NamedValues, Edge, SortedNames
from ..config import Config
from ..base import Configured

import core.scm as scm
from core import Env, Distributions


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

        raw = Batch.from_sample(self.named_tensors(action))

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

    def __getitem__(self, key: Edge):
        name_in, name_out = key
        temp = self.__causal_weight_action[name_out] / self.env.num_s
        try:
            return self.__causal_weights[name_out][name_in] + temp
        except KeyError:
            return temp

    def who_cause(self, name: str):
        return self.__causations[name]

    def infer(self, var: str, causal_states: Sequence[Any]):
        with torch.no_grad():
            weight = self.__attns[var]
            emb_a = self.__embs_a[var]
            caus = self.__causations[var]
            states = []
            for pa, state in zip(caus, causal_states):
                raw = self.v(pa).tensor(state, self.device)
                raw = raw.unsqueeze_(0)
                s = self.raw2input(pa, raw)
                states.append(self.network.encoder.forward(pa, s))
            states = self.T.safe_stack(states, (1, self.dims.variable_encoding))
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
