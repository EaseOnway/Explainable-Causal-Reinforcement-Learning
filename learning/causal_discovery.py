from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import utils
from core.buffer import Buffer
from core.taskinfo import TaskInfo


def discover(buffer: Buffer, taskinfo: TaskInfo, thres=0.05, log=True
             ) -> Dict[str, List[str]]:
    pa_dic: Dict[str, List[str]] = {
        key: [] for key in taskinfo.out_state_keys | taskinfo.outcomes_keys}

    data = buffer.read()

    for i in taskinfo.in_state_keys | taskinfo.action_keys:
        for j in taskinfo.out_state_keys | taskinfo.outcomes_keys:
            p: float = utils.fcit_test(
                data[i].reshape(len(buffer), -1),
                data[j].reshape(len(buffer), -1))  # type: ignore
            if p <= thres or np.isnan(p):
                pa_dic[j].append(i)  # biuld edge
            if log:
                print("independent test (%s, %s) done, p-value = %.5f " %
                      (i, j, p))
    if log:
        print('-------------------discovered-causal-graph---------------------')
        for name, parents in pa_dic.items():
            print(f"({', '.join(parents)}) --> {name}")

    return pa_dic  # type: ignore


def update(old: Dict[str, List[str]], buffer: Buffer,
          *edges: Tuple[str, str], thres=0.05, inplace=True,
           showinfo=True):
    if inplace:
        new = old
    else:
        new = {k: list(v) for k, v in old.items()}
    
    data = buffer.read()
    for i, j in edges:
        try:
            new[j].remove(i)
        except ValueError:
            pass

        p: float = utils.fcit_test(
            data[i].reshape(len(buffer), -1),
            data[j].reshape(len(buffer), -1))  # type: ignore
        if p <= thres or np.isnan(p):
            new[j].append(i)  # biuld edge
        if showinfo:
            print("independent test (%s, %s) done, p-value = %.5f " %
                (i, j, p))

    if showinfo:
        print('-------------------discovered-causal-graph---------------------')
        for name, parents in new.items():
            print(f"({', '.join(parents)}) --> {name}")
    return new
