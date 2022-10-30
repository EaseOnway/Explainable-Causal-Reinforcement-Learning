from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

import utils
from core.data import Buffer
from core.env import Env


ParentDict = Dict[str, Set[str]]


def discover(buffer: Buffer, env: Env, thres=0.05, log=True
             ) -> ParentDict:
    pa_dic: ParentDict = {
        key: set() for key in env.names_outputs}

    data = buffer.read_all()

    for i in env.names_inputs:
        for j in env.names_outputs:
            p: float = utils.fcit_test(
                data[i].reshape(len(buffer), -1),
                data[j].reshape(len(buffer), -1))  # type: ignore
            if p <= thres or np.isnan(p):
                pa_dic[j].add(i)  # biuld edge
            if log:
                print("independent test (%s, %s) done, p-value = %.5f " %
                      (i, j, p))
    if log:
        print('-------------------discovered-causal-graph---------------------')
        for name, parents in pa_dic.items():
            print(f"({', '.join(parents)}) --> {name}")

    return pa_dic  # type: ignore


def update(old: ParentDict, buffer: Buffer,
          *edges: Tuple[str, str], thres=0.05, inplace=True,
           showinfo=True):
    if inplace:
        new: ParentDict = old
    else:
        new = {k: v.copy() for k, v in old.items()}
    
    data = buffer.read_all()
    for i, j in edges:
        try:
            new[j].remove(i)
        except ValueError:
            pass

        p: float = utils.fcit_test(
            data[i].reshape(len(buffer), -1),
            data[j].reshape(len(buffer), -1))  # type: ignore
        if p <= thres or np.isnan(p):
            new[j].add(i)  # biuld edge
        if showinfo:
            print("independent test (%s, %s) done, p-value = %.5f " %
                (i, j, p))

    if showinfo:
        print('-------------------discovered-causal-graph---------------------')
        for name, parents in new.items():
            print(f"({', '.join(parents)}) --> {name}")
    return new
