from typing import Any, Dict, List, Optional, Set, Tuple, final
import abc
from utils import Shaping
import core.vtype as vtype
from enum import Enum


class EnvInfo:

    def __init__(self):
        self.action_names: Set[str] = set()
        self.state_names: Set[str] = set()
        self.outcome_names: Set[str] = set()
        self.vtypes: Dict[str, vtype.VType] = {}

    def _var(self, name: str, vtype: vtype.VType):
        if name in self.vtypes:
            raise ValueError(f"'{name}' already exists")
        self.vtypes[name] = vtype

    def action(self, name: str, vtype: vtype.VType):
        self._var(name, vtype)
        self.action_names.add(name)

    def state(self, name: str, vtype: vtype.VType):
        name_next = Env.name_next(name)
        self._var(name, vtype)
        self._var(name_next, vtype)
        self.state_names.add(name)

    def outcome(self, name: str, vtype: vtype.VType):
        self._var(name, vtype)
        self.outcome_names.add(name)


_NamedValues = Dict[str, Any]


class Env(abc.ABC):
    @staticmethod
    @final
    def name_next(name: str):
        return name + '\''
    
    def __str__(self):
        return type(self).__name__

    def __init__(self, info: EnvInfo):
        self.info = info
        self.__names_a = tuple(sorted(info.action_names))
        self.__names_s = tuple(sorted(info.state_names))
        self.__names_next_s = tuple(self.name_next(name) for name in self.__names_s)
        self.__nametuples_s = tuple(zip(self.__names_s, self.__names_next_s))
        self.__names_o = tuple(sorted(info.outcome_names))
        self.__names_inputs = self.__names_s + self.__names_a
        self.__names_outputs = self.__names_o + self.__names_next_s
        self.__num_a = len(self.__names_a)
        self.__num_s = len(self.__names_s)
        self.__num_o = len(self.__names_o)
        self.__action_id_map = {k: i for i, k in enumerate(self.__names_a)}
        self.__state_id_map = {k: i for i, k in enumerate(self.__names_s)}
        self.__state_id_map.update({k: i for i, k in enumerate(self.__names_next_s)})
        self.__outcome_id_map = {k: i for i, k in enumerate(self.__names_o)}
        self.__vtypes: Dict[str, vtype.VType] = info.vtypes
    
        self.reset()
    
    def reset(self, *args, **kargs):
        ''' initialiize the current state
        '''
    
        self.__current_state = self.init(*args, **kargs)
    

    def step(self, action: _NamedValues) -> Tuple[_NamedValues,
                                                  float, bool, Any]:
        ''' input acitons, gain outcomes, and update states. if done, reset.
            return
            - a dict comprising the transition (s, a, o, s')
            - reward: float
            - done (bool)
            - other information (Any)
        '''

        transition = self.__current_state.copy()
        transition.update(action)
        out, info = self.transit(transition)
        transition.update(out)
        done = self.done(transition, info)
        reward = self.reward(transition)
        
        if not done:
            self.__current_state = {name: transition[Env.name_next(name)]
                                    for name in self.names_s}
        else:
            self.reset()

        return transition, reward, done, info

    @abc.abstractmethod
    def init(self, *args, **kargs) -> _NamedValues:
        '''
        initialiize the state dict
        '''
        raise NotImplementedError

    @abc.abstractmethod    
    def transit(self, states_and_actions: _NamedValues
                ) -> Tuple[_NamedValues, Any]:
        '''
        return:
        - next states and outcomes (dict)
        - other information (Any)
        '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def done(self, transition: _NamedValues, info: Any) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self, transition: _NamedValues) -> float:
        raise NotImplementedError
    
    @abc.abstractmethod
    def random_action(self) -> _NamedValues:
        raise NotImplementedError

    @property
    @final
    def current_state(self):
        return self.__current_state

    @property
    @final
    def names_a(self):
        return self.__names_a
    
    @property
    @final
    def names_s(self):
        return self.__names_s
    
    @property
    @final
    def names_next_s(self):
        return self.__names_next_s
    
    @property
    @final
    def nametuples_s(self):
        return self.__nametuples_s
    
    @property
    @final
    def names_inputs(self):
        return self.__names_inputs

    @property
    @final
    def names_outputs(self):
        return self.__names_outputs
    
    @property
    @final
    def names_o(self):
        return self.__names_o

    @property
    @final
    def num_s(self):
        return self.__num_s
    
    @property
    @final
    def num_a(self):
        return self.__num_a
    
    @property
    @final
    def num_o(self):
        return self.__num_o
    
    @final
    def name_s(self, i: int, next=False):
        if next:
            return self.__names_next_s[i]
        else:
            return self.__names_s[i]
    
    @final
    def idx_s(self, name: str):
        return self.__state_id_map[name]
    
    @final
    def name_a(self, i: int):
        return self.__names_a[i]
    
    @final
    def idx_a(self, name: str):
        return self.__action_id_map[name]
    
    @final
    def name_o(self, i: int):
        return self.__names_o[i]
    
    @final
    def idx_o(self, name: str):
        return self.__outcome_id_map[name]
    
    @final
    def info_s(self, i: int):
        return self.__vtypes[self.name_s(i)[0]]
    
    @final
    def info_a(self, i: int):
        return self.__vtypes[self.name_a(i)]
    
    @final
    def info_o(self, i: int):
        return self.__vtypes[self.name_o(i)]
    
    @final
    def var(self, name: str):
        return self.__vtypes[name]
