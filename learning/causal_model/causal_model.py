from typing import Optional
import torch
import numpy as np

from .causal_net import CausalNet
from core import EnvInfo
from ..data import Batch, Transitions, Tag
from ..buffer import Buffer
from ..base import Configured

from utils.typings import NamedTensors


class CausalModel(Configured):
    def __init__(self, net: CausalNet, truth_buffer: Buffer,
                 n: int, max_traj_len: Optional[int] = None):
        super().__init__(net.config)
        self.__true_env = net.env
        self.init_buffer = self.__get_init_buffer(truth_buffer)
        self.n = n
        self.model = net
        self.max_traj_len = max_traj_len
    
    def __get_init_buffer(self, buffer: Buffer):
        initiated = buffer.transitions[:].initiated.cpu()
        transitions = buffer.transitions[initiated]
        new_buffer = Buffer(self.config, max_size=transitions.n)
        new_buffer.append(transitions)
        assert torch.all(new_buffer.transitions[:].initiated)
        return new_buffer

    def __transit(self, states_and_actions: Batch):
        out = self.model.forward(states_and_actions)
        return out.sample().kapply(self.label2raw)

    def __complete_transition(self, transition: Batch):
        r = np.empty(transition.n, float)
        
        code = np.zeros(transition.n, np.int32)
        code[self.__initiated] |= Tag.INITIATED.mask
        
        arrays = self.as_numpy(transition)
        self.__i_step += 1

        for i in range(transition.n):
            sample = {k: a[i] for k, a in arrays.items()}
            r[i] = self.__true_env.reward(sample)
            terminated = self.__true_env.terminated(sample)
            if terminated:
                code[i] |= Tag.TERMINATED.mask
                self.__i_step[i] = 0

        if self.max_traj_len is not None:
            truncated: np.ndarray = (self.__i_step >= self.max_traj_len)
            code[truncated] |= Tag.TRUNCATED.mask
            self.__i_step[truncated] = 0

        return Transitions(transition.data, self.T.a2t(r), self.T.a2t(code))

    def reset(self):
        batch = self.init_buffer.sample_batch(self.n)
        self.__current_state = batch.select(self.env.names_s)
        self.__i_step = np.zeros(self.n, dtype=int)
        self.__initiated = np.ones(self.n, dtype=bool)
    
    @property
    def current_state(self):
        return self.__current_state

    def step(self, action: Batch):
        tran = Batch(action.n)
        tran.update(self.__current_state)
        tran.update(action)
        tran.update(self.__transit(tran))
        tran = self.__complete_transition(tran)
        self.__current_state = self.shift_states(tran)

        # reset
        done = tran.done
        n_done = int(torch.count_nonzero(done))
        reset = self.init_buffer.sample_batch(n_done)
        for k in self.__current_state.keys():
            self.__current_state[k] = torch.clone(self.__current_state[k])
            self.__current_state[k][done] = reset[k]
        self.__initiated = self.T.t2a(done, dtype=bool)

        return tran
