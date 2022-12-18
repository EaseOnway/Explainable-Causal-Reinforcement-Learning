from typing import Any, Dict, Optional, Sequence, Tuple, Union, Iterable, Set

import numpy as np
import torch

from ..env_model import AttnCausalModel
from core import Batch, Env
from utils.visualize import plot_digraph
from utils.typings import NamedTensors, NamedValues, Edge, SortedNames, ParentDict
from ..config import Config

import core.scm as scm


class ActionEffect:
    def __init__(self, env: Env, causal_graph: ParentDict, 
                 transition: Env.Transition, attn: torch.Tensor,
                 interest: Optional[Set[str]] = None):

        self.__env = env
        self.action = env.action_of(transition.variables)
        self.interest = interest
        self.causal_graph = causal_graph
        self.attn_weights = {env.names_input[j]:
            {env.names_output[i]: float(attn[j, i])
                for i in range(env.num_input)}
        for j in range(env.num_output)}
        
    def __getitem__(self, key: Edge):
        i, j = key
        if self.interest is not None and \
                len(self.interest & set(self.causal_graph[j])) == 0:
            return 0.
        return self.attn_weights[j][i]

    def who_cause(self, out: str):
        return tuple(i for i in self.causal_graph[out] if i in self.__env.names_s)
