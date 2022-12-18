from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import joblib

import utils
from utils.typings import ParentDict, NamedArrays, SortedNames

from .buffer import Buffer
from core.env import Env


def _concat_data(data: NamedArrays, names: Iterable[str]) -> np.ndarray:
    to_cat = [data[name] for name in names]
    return np.concatenate(to_cat, axis=1)


def _test(edge: Tuple[str, str], data: NamedArrays, input_names: SortedNames):
    i, j = edge
    cond = _concat_data(data,
        (name for name in input_names if name != i))
    p = utils.fcit_test(data[i], data[j], cond)
    assert isinstance(p, float)
    print("\t(%s) caused by (%s) with assurrance %.5f " % (j, i, 1 - p))
    return p


def discover(data: NamedArrays, env: Env, thres=0.05, showinfo=True,
             n_jobs=1) -> ParentDict:
    pa_dic = {key: set() for key in env.names_output}
    
    print('starting causal discovery')
    edges = [(i, j) for j in env.names_output for i in env.names_input]

    p_values = joblib.Parallel(n_jobs)(
        joblib.delayed(_test)(edge, data, env.names_input)
        for edge in edges)
    assert p_values is not None

    for (i, j), p in zip (edges, p_values):
        dependent = p <= thres # or np.isnan(p)
        if dependent:
            pa_dic[j].add(i)
        # if showinfo:
        #    print("\tcaused by (%s) with assurrance %.5f " % (i, 1 - p))
    
    if showinfo:
        print('-------------------discovered-causal-graph---------------------')
        for name, parents in pa_dic.items():
            print(f"({', '.join(parents)}) --> {name}")
        print('---------------------------------------------------------------')
    
    return {k: tuple(sorted(pa)) for k, pa in pa_dic.items()}
  