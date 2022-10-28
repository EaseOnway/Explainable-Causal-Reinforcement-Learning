from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .networks.causal_model.inferrer import Inferrer
from .networks.causal_model import CausalNet
from core import Batch
import utils.tensorfuncs as T
from utils.visualize import plot_digraph
from .config import Configured, Config

import core.scm as scm


class ActionEffect(Configured):
    def __init__(self, network: CausalNet,
                 action_datadic: Dict[str, np.ndarray],
                 attn_thres=0.):
        super().__init__(network.config)
        self.network = network

        # compute:
        # 1. the encodings of action variables
        # 2. the action-effect embedding of each structural function
        # 3. the attention scores of each structrual function

        batch = Batch.from_sample(action_datadic)
        with torch.no_grad():
            actions = network.encoder.forward_all(network.a2t(batch))
            a_embs: Dict[str, torch.Tensor] = {}
            causations: Dict[str, Tuple[str, ...]] = {}
            weights: Dict[str, torch.Tensor] = {}
            for var in self.env.names_outputs:
                parents_a = network.parent_dic_a[var]
                parents_s = network.parent_dic_s[var]
                actions_pa = T.safe_stack([actions[pa] for pa in parents_a],
                                          (1, self.config.dims.action_encoding),
                                          **self.torchargs)
                inferrer = self.network.inferrers[var]
                a_emb = inferrer.aggregator.forward(actions_pa)
                kstates = self.network.k_model.forward(parents_s)
                attn = inferrer.get_attn_scores(a_emb, kstates)
                
                a_embs[var] = a_emb
                if len(parents_s) > 0:
                    selected = (attn >= attn_thres/len(parents_s))
                    weight = attn[selected]
                    causation = tuple(s for s, sel in zip(parents_s, selected) if sel)
                else:
                    weight = attn
                    causation = parents_s
                causations[var] = causation
                weights[var] = weight
        
        self.actions = actions
        self.__embs_a = a_embs
        self.__causations = causations
        self.__weights = weights
    

    def infer(self, var: str, causal_states: Sequence[Any]):
        with torch.no_grad():
            weight = self.__weights[var]
            emb_a = self.__embs_a[var]
            caus = self.__causations[var]
            states = []
            for pa, s in zip(caus, causal_states):
                v = self.v(pa)
                s = np.array(s, v.dtype).reshape(1, *v.shape)
                states.append(self.network.encoder.forward(
                    pa, T.a2t(s, **self.torchargs)))
            states = T.safe_stack(states, (1, self.dims.state_encoding),
                                  **self.torchargs)
            inferrer = self.network.inferrers[var]
            out = inferrer.attn_infer(weight, emb_a, states)
            out = inferrer.decoder(out)
            out = inferrer.predict(out)
            out = np.squeeze(out, axis=0)
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

    def plot(self, format='png'):
        return plot_digraph(
            self.config.keys_out, self.__causations,  # type: ignore
            format=format)
