from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .networks.inferrer import Inferrer
from .networks import CausalNet
import utils as u

import core.scm as scm


class ActionEffect:
    def __init__(self, network: CausalNet,
                 action_datadic: Dict[str, np.ndarray],
                 attn_thres=0.):
        self.network = network
        self._cfg = network.config

        # convert to batch data, where batchsize = 1
        action_datadic = {var: np.expand_dims(value, 0) for var, value
                          in action_datadic.items()}

        # compute:
        # 1. the encodings of action variables
        # 2. the action-effect embedding of each structural function
        # 3. the attention scores of each structrual function
        with torch.no_grad():
            actions = network.action_encoder.forward_all(action_datadic)
            a_embs: Dict[str, torch.Tensor] = {}
            causations: Dict[str, Tuple[str, ...]] = {}
            weights: Dict[str, torch.Tensor] = {}
            for var in self._cfg.outkeys:
                parents_a = network.parent_dic_a[var]
                parents_s = network.parent_dic_s[var]
                if len(parents_a) > 0:
                    actions_pa = torch.stack([actions[pa] for pa in parents_a])
                else:
                    actions_pa = torch.zeros((0, 1, self._cfg.dims.a),
                                            **self._cfg.torchargs)
                
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
        
        self.__actions = actions
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
                v = self._cfg.var(pa)
                shape = (len(v.shape),) if v.categorical else v.shape
                s = np.array(s, v.dtype).reshape(1, *shape)
                states.append(self.network.state_encoder.forward(pa, s))
            states = u.safe.stack(states, (1, self._cfg.dims.s), **self._cfg.torchargs)
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
        for var in self._cfg.task.in_state_keys:
            m[var] = scm.ExoVar(name=var)
        for var in self._cfg.outkeys:
            m[var] = scm.EndoVar(
                [m[pa] for pa in self.__causations[var]],
                self.__get_causal_eq(var),
                name = var,
            )
        return m

    def print_info(self):
        for var in self._cfg.outkeys:
            print(f"exogenous variable '{var}':")
            if len(self.__causations[var]) > 0:
                for cau, w in zip(self.__causations[var], self.__weights[var]):
                    print(f"\tcaused by {cau} with weight {float(w)}")
            else:
                print("\tno causations")

    def plot(self, format='png'):
        return u.plot_digraph(self._cfg.outkeys, self.__causations,  # type: ignore
                            format=format)
