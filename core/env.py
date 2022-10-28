from typing import Any, Dict, List, Optional, Set, Tuple, final
import numpy as np
import abc
from utils.shaping import *


class VarInfo:
    def __init__(self, shape: ShapeLike, dtype: type,
                 default: Optional[Any]):
        self.shape = as_shape(shape)
        self.size = get_size(shape)
        self.dtype = dtype
        self.default = default


class EnvInfo:
    def __init__(self):
        self.action_names: Set[str] = set()
        self.state_names: Set[Tuple[str, str]] = set()
        self.outcome_names: Set[str] = set()
        self.varinfos: Dict[str, VarInfo] = {}

    def var(self, name: str, shape: ShapeLike = (),
             dtype: type = float, default: Optional[Any] = None):
        if name in self.varinfos:
            raise ValueError(f"'{name}' already exists")
        self.varinfos[name] = VarInfo(shape, dtype, default)

    def action(self, name: str, shape: ShapeLike,
               onehot=True, dtype: type = np.uint8, 
               default: Optional[Any] = None):
        self.var(name, shape, dtype, default)
        self.action_names.add(name)

    def state(self, name: str, dtype: type = float,
              shape: ShapeLike = (), default: Optional[Any] = None):
        name_next = Env.name_next_step(name)
        self.var(name, shape, dtype, default)
        self.var(name_next, shape, dtype, default)
        self.state_names.add((name, name_next))

    def outcome(self, name: str, default: Optional[Any] = None):
        self.var(name, (), float, default)
        self.outcome_names.add(name)


_KArrays = Dict[str, np.ndarray]


class Env(abc.ABC):
    @staticmethod
    @final
    def name_next_step(name: str):
        return name + '\''


    def __init__(self, info: EnvInfo, 
                 task_weights: Optional[Dict[str, float]] = None):
        self.info = info
        self.__names_a = tuple(sorted(info.action_names))
        names_s = tuple(sorted(info.state_names))
        self.__names_s = tuple(name[0] for name in names_s)
        self.__names_s_next = tuple(name[1] for name in names_s)
        self.__names_inputs = self.__names_s + self.__names_a
        self.__names_outputs = self.__names_o + self.__names_s_next
        self.__names_o = tuple(sorted(info.outcome_names))
        self.__num_a = len(self.__names_a)
        self.__num_s = len(self.__names_s)
        self.__num_o = len(self.__names_o)
        
        self.__action_id_map = {k: i for i, k in enumerate(self.__names_a)}
        self.__state_id_map = {k[0]: i for i, k in enumerate(self.__names_s)}
        self.__state_id_map.update({k[1]: i for i, k in enumerate(self.__names_s)})
        self.__outcome_id_map = {k: i for i, k in enumerate(self.__names_o)}
        self.__varinfos: Dict[str, VarInfo] = info.varinfos
        
        self.__weights_o = np.zeros(self.num_o, dtype=float)
        if task_weights is not None:
            self.set_task(task_weights)
    
        self.reset()
    
    def reset(self):
        ''' initialiize the current state
        '''

        self.__current_state = self.init()
    

    def step(self, action: _KArrays) -> Tuple[_KArrays, float, bool, Any]:
        ''' input acitons, gain outcomes, and update states. if done, reset.
            return
            - a dict comprising the transition (s, a, o, s')
            - reward: float
            - done (bool)
            - other information (Any)
        '''

        transition: _KArrays = {}
        transition.update(action)
        out, info = self.transit(transition)
        for name in self.names_outputs:
            if name not in out:
                raise ValueError(f"{name} is missing")
        transition.update(out)
        done = self.done(transition, info)
        if not done:
            self.__current_state = {name: transition[Env.name_next_step(name)]
                                    for name in self.names_s}
        else:
            self.reset()
        reward = sum(self.weight_o(i) * transition[self.name_o(i)]
                     for i in range(self.num_o))
        return transition, reward, done, info

    @abc.abstractmethod
    def init(self) -> _KArrays:
        '''
        initialiize the current state dict
        '''
        raise NotImplementedError

    @abc.abstractmethod    
    def transit(self, states_and_actions: _KArrays
                ) -> Tuple[_KArrays, Any]:
        '''
        return:
        - next states and outcomes (dict)
        - other information (Any)
        '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def done(self, trainsition: _KArrays, info: Any) -> bool:
        raise NotImplementedError

    @final
    def set_task(self, weights: Dict[str, float]):
        self.__weights_o = np.zeros(self.num_o, dtype=float)
        for name, w in weights.items():
            self.__weights_o[self.idx_o(name)] = w

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
        return self.__names_s_next
    
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
            return self.__names_s_next[i]
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
    def weight_o(self, i: int):
        return self.__weights_o[i]
    
    @property
    @final
    def weights_o(self):
        return self.__weights_o
    
    @final
    def info_s(self, i: int):
        return self.__varinfos[self.name_s(i)[0]]
    
    @final
    def info_a(self, i: int):
        return self.__varinfos[self.name_a(i)]
    
    @final
    def info_o(self, i: int):
        return self.__varinfos[self.name_o(i)]
    
    @final
    def var(self, name: str):
        return self.__varinfos[name]
    
    @final
    def reward(self, outcomes: _KArrays):
        return sum(self.weight_o(i) * outcomes[self.name_o(i)]
                   for i in range(self.num_o))
