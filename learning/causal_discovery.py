from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import joblib

import utils
from utils.typings import ParentDict, NamedArrays, SortedNames

from .buffer import Buffer
from core.env import Env
from core.vtype import Categorical
from typing import Sequence
import numpy as np
import joblib
from utils import fcit_test

# def _test(i: int, xs: Sequence[np.ndarray], y: np.ndarray):
#     x = xs[i]
#     z = np.concatenate([xs[j] for j in range(len(xs)) if j != i], axis=1)
#     p = fcit_test(x, y, z)
#     print('.', end='')
#     return p
# 
# 
# def _test_rl(xs: Sequence[np.ndarray], ys: Sequence[np.ndarray], n_jobs=1):
#     edges = [(i, j) for i in range(len(xs)) for j in range(len(ys))]
#     print("Performing FCIT tests: ", end='')
#     p_values = joblib.Parallel(n_jobs=n_jobs)(
#         joblib.delayed(_test)(i, xs, ys[j]) for i, j in edges
#     )
#     print(' Done!')
#     assert p_values is not None
#     return {
#         (i, j): p
#         for (i, j), p in zip (edges, p_values)
#     }

# def discover(data: NamedArrays, env: Env, thres=0.05, showinfo=True,
#             n_jobs=1) -> ParentDict:
#     pa_dic = {key: set() for key in env.names_output}
# 
#     print('starting causal discovery')
#     p_values = _test_rl([data[name] for name in env.names_input],
#                         [data[name] for name in env.names_output],
#                         n_jobs)
# 
# 
#     for (i, j), p in p_values.items():
#         i = env.names_input[i]
#         j = env.names_output[j]
#         dependent = p <= thres # or np.isnan(p)
#         if dependent:
#             pa_dic[j].add(i)
#         # if showinfo:
#         #    print("\tcaused by (%s) with assurrance %.5f " % (i, 1 - p))
# 
#     if showinfo:
#         print('-------------------discovered-causal-graph---------------------')
#         for name, parents in pa_dic.items():
#             print(f"({', '.join(parents)}) --> {name}")
#         print('---------------------------------------------------------------')
# 
#     return {k: tuple(sorted(pa)) for k, pa in pa_dic.items()}
#     

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

def discover_aim(data: NamedArrays, env: Env, thres=0.05, showinfo=True, n_jobs=1):
    if len(env.names_a) != 1:
        raise ValueError("the environment can only contain one categorical action variable.")

    name_a = env.names_a[0]
    vtype = env.var(name_a)
    if not isinstance(vtype, Categorical):
        raise ValueError("the action is not categorical.")
    
    k = vtype.k
    out: List[ParentDict] = []

    edges = [(i, j) for j in env.names_output for i in env.names_s]

    for a in range(k):
        print("------------action #%d-------------" % a)

        temp: np.ndarray = (data[name_a][:, a].astype(bool))
        data_a = {
            name: value[temp] for name, value in data.items()
            if name != name_a
        }
    
        p_values = joblib.Parallel(n_jobs)(
            joblib.delayed(_test)(edge, data_a, env.names_s)
            for edge in edges)
        assert p_values is not None
        
        pa_dic = {key: set() for key in env.names_output}
        for (i, j), p in zip (edges, p_values):
            dependent = p <= thres # or np.isnan(p)
            if dependent:
                pa_dic[j].add(i)
        pa_dic = {k: tuple(sorted(pa)) for k, pa in pa_dic.items()}
        out.append(pa_dic)
    
    return out
