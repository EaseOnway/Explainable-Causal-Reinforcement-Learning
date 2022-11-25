from typing import Union, Optional
import torch
import numpy as np

from .causal_net import CausalNet
from core import Env
from core import Batch, Transitions, Tag
from ..buffer import Buffer
from ..base import Configured

from utils.typings import NamedTensors, NamedValues
import utils


class CausalModel(Configured):
    def __init__(self, net: CausalNet, truth_buffer: Buffer,
                 max_trlen: Optional[int] = None):
        super().__init__(net.config)
        self.__true_env = net.env
        self.init_buffer = self.__get_init_buffer(truth_buffer)
        self.model = net
        self.max_trlen = max_trlen
    
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

        if self.max_trlen is not None:
            truncated: np.ndarray = (self.__i_step >= self.max_trlen)
            code[truncated] |= Tag.TRUNCATED.mask
            self.__i_step[truncated] = 0

        return Transitions(transition.data, self.T.a2t(r), self.T.a2t(code))

    def reset(self, arg: Union[Batch, int]):
        if isinstance(arg, int):
            n = arg
            batch = self.init_buffer.sample_batch(n)
        else:
            batch = arg
            n = batch.n
                
        n = batch.n
        self.__current_state = batch.select(self.env.names_s)
        self.__i_step = np.zeros(n, dtype=int)
        self.__initiated = np.ones(n, dtype=bool)
    
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


class SimulatedEnv(Env):
    def __init__(self, net: CausalNet, init_state: NamedValues,
                 mode=False):
        super().__init__(net.env._def)
        self.mode = mode
        self.__true_env = net.env
        self.__net = net
        self.init_state = init_state.copy()

    def init_episode(self) -> NamedValues:
        return self.init_state
    
    def transit(self, actions: NamedValues):
        sa = utils.Collections.merge_dic(self.current_state, actions)
        return self.__net.simulate(sa, self.mode), {}

    def terminated(self, transition: NamedValues) -> bool:
        return self.__true_env.terminated(transition)
    
    def random_action(self) -> NamedValues:
        return self.__true_env.random_action()