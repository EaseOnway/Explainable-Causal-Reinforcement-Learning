from typing import Any, Dict, Sequence, Optional, Set, Tuple, final, Callable, List
import abc
import core.vtype as vtype
from utils.typings import NamedValues, SortedNames
from argparse import ArgumentParser


class Env(abc.ABC):

    class _Rewarder:
        def __init__(self, source: Sequence[str],
                     func: Callable[..., float]):
            self.__source = tuple(source)
            self.__func = func
        
        def __call__(self, variables: NamedValues):
            values = [variables[k] for k in self.__source]
            return self.__func(*values)
        
        @property
        def source(self):
            return self.__source

    class Definition:

        def __init__(self):
            self.action_names: Set[str] = set()
            self.state_names: Set[str] = set()
            self.outcome_names: Set[str] = set()
            self.vtypes: Dict[str, vtype.VType] = {}
            self.rewarders: Dict[str, Env._Rewarder] = {}

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
        
        def reward(self, label: str, source: Sequence[str],
                   func: Callable[..., float]):
            self.rewarders[label] = Env._Rewarder(source, func)

    class Transition:
        def __init__(self, variables: NamedValues,
                     reward: float, terminated: bool, **info):
            self.variables = variables
            self.reward = reward
            self.terminated = terminated
            self.info = info

    @staticmethod
    @final
    def name_next(name: str):
        return name + '\''
    
    def __str__(self):
        return type(self).__name__

    @final
    def __init__(self, args):

        _def = self.define(args)
        self.__def = _def
        self.__names_a: SortedNames = tuple(sorted(_def.action_names))
        self.__names_s: SortedNames = tuple(sorted(_def.state_names))
        self.__names_next_s: SortedNames = tuple(self.name_next(name) for name in self.__names_s)
        self.__nametuples_s = tuple(zip(self.__names_s, self.__names_next_s))
        self.__names_o: SortedNames = tuple(sorted(_def.outcome_names))
        self.__names_inputs: SortedNames = tuple(sorted(self.__names_s + self.__names_a))
        self.__names_outputs: SortedNames = tuple(sorted(self.__names_o + self.__names_next_s))
        self.__names_all: SortedNames = tuple(sorted(self.__names_inputs + self.__names_outputs))
        self.__num_a = len(self.__names_a)
        self.__num_s = len(self.__names_s)
        self.__num_o = len(self.__names_o)
        self.__vtypes: Dict[str, vtype.VType] = _def.vtypes
        self.__current_state: NamedValues
        self.__t: int

    @property
    @final
    def definition(self):
        return self.__def
    
    @property
    def rewarders(self):
        return self.__def.rewarders

    @classmethod
    @abc.abstractmethod
    def init_parser(cls, parser: ArgumentParser):
        raise NotImplementedError

    @abc.abstractmethod
    def define(self, args) -> Definition:
        raise NotImplementedError
    
    @abc.abstractmethod
    def launch(self):
        '''set up core environment of gym, pysc2, etc'''
        raise NotImplementedError

    def reset(self, *args, **kargs):
        ''' initialiize the current state
        '''
        
        self.__t = 0
        self.__current_state = self.init_episode(*args, **kargs)
    
    @property
    @final
    def current_state(self):
        try:
            return self.__current_state
        except AttributeError:
            self.reset()
            return self.__current_state

    @property
    def t(self):
        return self.__t

    def step(self, action: NamedValues) -> Transition:
        ''' input acitons, gain outcomes, and update states. if done, reset.
            return
            - a dict comprising the transition (s, a, o, s')
            - reward: float
            - done (bool)
            - other information (Any)
        '''

        variables: NamedValues = action.copy()
        variables.update(self.current_state)
        variables.update(action)

        out, info = self.transit(action)

        variables.update(out)

        terminated = self.terminated(variables)
        reward = self.reward(variables)
        
        if not terminated:
            self.__current_state = self.state_shift(variables)
            self.__t += 1
        else:
            self.reset()

        return Env.Transition(variables, reward, terminated, **info)
    
    @abc.abstractmethod
    def init_episode(self, *args, **kargs) -> NamedValues:
        '''
        initialiize the state dict
        '''
        raise NotImplementedError

    @abc.abstractmethod    
    def transit(self, actions: NamedValues) -> Tuple[NamedValues, Any]:
        '''
        return:
        - next states and outcomes (dict)
        - other information (Any)
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def random_action(self) -> NamedValues:
        raise NotImplementedError
    
    @abc.abstractmethod
    def terminated(self, variables: NamedValues) -> bool:
        raise NotImplementedError

    @final
    def reward(self, variables: NamedValues) -> float:
        r = 0.
        for rewarder in self.rewarders.values():
            r += rewarder(variables)
        return r

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
    def names_o(self):
        return self.__names_o

    @property
    @final
    def names_input(self):
        return self.__names_inputs

    @property
    @final
    def names_output(self):
        return self.__names_outputs

    @property    
    def names_all(self):
        return self.__names_all

    @final
    def has_name(self, name: str):
        return name in self.__vtypes

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
    
    @property
    @final
    def num_input(self):
        return self.__num_a + self.__num_s
    
    @property
    @final
    def num_output(self):
        return self.__num_o + self.__num_s

    @final
    def var(self, name: str):
        return self.__vtypes[name]

    def state_of(self, variables: NamedValues) -> NamedValues:
        return {k: variables[k] for k in self.__names_s}

    def action_of(self, variables: NamedValues) -> NamedValues:
        return {k: variables[k] for k in self.__names_a}

    def next_state_of(self, variables: NamedValues) -> NamedValues:
        return {k: variables[k] for k in self.__names_next_s}

    def outcomes_of(self, variables: NamedValues) -> NamedValues:
        return {k: variables[k] for k in self.__names_o}
    
    def state_shift(self, variables: NamedValues) -> NamedValues:
        return {s: variables[s_] for s, s_ in self.nametuples_s}

    def texts(self, variables: NamedValues):
        return {k: self.__vtypes[k].text(v)
                for k, v in variables.items()}

    def get_full_graph(self):
        return {name_out: self.names_input
                for name_out in self.names_output}
