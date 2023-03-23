from typing import Any, Dict, Optional, Sequence, Tuple, Union, Iterable, Set

import numpy as np
import torch

from ..env_model.modules import Inferrer
from ..env_model import CausalEnvModel
from core import Batch
from utils.visualize import plot_digraph
from utils.typings import NamedTensors, NamedValues, Edge, SortedNames
from ..config import Config
from ..base import RLBase

import core.scm as scm


class ActionEffect(RLBase):
    def __init__(self, network: CausalEnvModel, action: NamedValues,
                 interest: Optional[Set[str]] = None):

        super().__init__(network.context)

        self.network = network
        self.action = action
        
        self.interest = interest

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
            action_enc = network.encoder.forward(batch)
            causations: Dict[str, Tuple[str]] = {}
            attns: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            causal_weights: Dict[str, Dict[str, float]] = {}
            causal_weight_action: Dict[str, float] = {}

            for var in self.env.names_output:
                parents_a = network.parent_dic_a[var]
                parents_s = network.parent_dic_s[var]
                actions_pa = self.T.safe_stack([action_enc[pa] for pa in parents_a],
                                               (1, self.config.dims.variable_encoding))
                inferrer = self.network.inferrers[var]
                a_emb = inferrer.aggregator.forward(actions_pa)
                kstates = self.network.k_model.forward(parents_s)
                attn = inferrer.get_attn_scores(a_emb, kstates)

                causations[var] = parents_s
                attns[var] = attn

                weight_s = self.T.t2a(attn[0].view(len(parents_s)))
                weight_a = float(attn[1].view(()))

                if self.interest is not None and \
                        len(set(parents_a) & self.interest) == 0:
                    causal_weights[var] = {i: 0. for i in parents_s}
                    causal_weight_action[var] = 0.
                else:
                    causal_weights[var] = {i: float(w) for i, w
                                            in zip(parents_s, weight_s)}
                    causal_weight_action[var] = float(weight_a)

        self.__action_enc = action_enc
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

    def who_cause(self, name: str, threshold: Optional[float] = None) -> Tuple[str, ...]:
        if threshold is None:
            return self.__causations[name]
        else:
            weights = self.__causal_weights[name]
            return tuple(pa for pa in self.__causations[name] if weights[pa] >= threshold)

    def graph(self, threshold: float):
        return {name: self.who_cause(name, threshold)
                for name in self.env.names_output}

    def print_info(self):
        for var in self.env.names_output:
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
