import torch
import numpy as np

from .causal_net import CausalNet
from core import EnvInfo
from ..data import Batch, Transitions
from ..buffer import Buffer
from ..base import Configured

from utils.typings import NamedTensors


class CausalModel(Configured):
    def __init__(self, net: CausalNet, buffer: Buffer, n: int):
        super().__init__(net.config)

        self.__true_env = net.env
        self.buffer = buffer
        self.n = n
        self.model = net

        self.reset()

    def __transit(self, states_and_actions: Batch):
        out = self.model.forward(states_and_actions)
        return out.sample().kapply(self.label2raw)

    def __reward_and_code(self, transition: Batch):
        r = np.empty(transition.n, float)
        code = np.empty(transition.n, np.uint8)

        arrays = self.as_numpy(transition)

        for i in range(transition.n):
            sample = {k: a[i] for k, a in arrays.items()}
            r[i] = self.__true_env.reward(sample)
            done = self.__true_env.done(sample)
            code[i] = Transitions.get_code(done, False)

        return self.T.a2t(r), self.T.a2t(code)

    def reset(self):
        batch = self.buffer.sample_batch(self.n)
        self.__current_state = batch.select(self.env.names_s)
    
    @property
    def current_state(self):
        return self.__current_state

    def step(self, action: Batch):
        tran = Batch(action.n)
        tran.update(self.__current_state)
        tran.update(action)
        tran.update(self.__transit(tran))
        r, code = self.__reward_and_code(tran)

        tran = Transitions(tran.data, r, code)
        self.__current_state = self.shift_states(tran)

        done = tran.done
        n_done = int(torch.count_nonzero(done))
        reset = self.buffer.sample_batch(n_done)

        for k in self.__current_state.keys():
            self.__current_state[k] = torch.clone(self.__current_state[k])
            self.__current_state[k][done] = reset[k]

        return tran
