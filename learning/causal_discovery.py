from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import utils
from utils.typings import ParentDict, NamedArrays, Edge

from .buffer import Buffer
from core.env import Env


def _concat_data(data: NamedArrays, names: Iterable[str]) -> np.ndarray:
    to_cat = [data[name] for name in names]
    return np.concatenate(to_cat, axis=1)


def update(old: ParentDict, data: NamedArrays, env: Env, target: str, 
           thres=0.05, inplace=True, showinfo=True):
    assert target in env.names_outputs

    if inplace:
        new: ParentDict = old
    else:
        new = old.copy()

    pa = set()

    if showinfo:
        print(f"finding causations of '{target}':")

    def _test(i: str):
        nonlocal pa, data

        cond = _concat_data(data,
            (name for name in env.names_inputs if name != i))
        p = utils.fcit_test(data[i], data[target], cond)
        assert isinstance(p, float)
        dependent = p <= thres # or np.isnan(p)
        if dependent:
            pa.add(i)
        if showinfo:
            print("\tcaused by (%s) with assurrance %.5f " % (i, 1 - p))

    # first, do independent tests with states, conditioned on actions
    for i in env.names_inputs:
        _test(i)
    
    print(f"Result: ({', '.join(pa)}) --> {target}")
    
    new[target] = pa
    return new


def discover(data: NamedArrays, env: Env, thres=0.05, showinfo=True
             ) -> ParentDict:
    pa_dic: ParentDict = {
        key: set() for key in env.names_outputs}

    for j in env.names_outputs:
        update(pa_dic, data, env, j, thres, inplace=True, showinfo=showinfo)
    
    if showinfo:
        print('-------------------discovered-causal-graph---------------------')
        for name, parents in pa_dic.items():
            print(f"({', '.join(parents)}) --> {name}")
        print('---------------------------------------------------------------')
    
    return pa_dic


