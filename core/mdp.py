from typing import Any, Callable, Dict, Sequence, Optional

import numpy as np

from .scm import ExoVar, EndoVar, StructrualCausalModel
from .env import EnvInfo, Env


class CausalMdp(Env):

    def __init__(self, envinfo: EnvInfo):
        super().__init__(envinfo)
        self.__scm = StructrualCausalModel()

        self.__undefined = set(self.names_outputs)
        for name in self.names_inputs:
            self.__scm.add(ExoVar(name, self.var(name)))
    
    @property
    def scm(self):
        return self.__scm
        
    def define(self, name: str, parents: Sequence[str], eq: Callable):
        if name in self.names_s:
            name = self.name_next(name)
        if name not in self.__undefined or name not in self.names_outputs:
            raise ValueError(f"{name} is not a endogenous variable of the environment"
                             " or has already been defined.")
        self.__scm.add(EndoVar([self.__scm[pa] for pa in parents], eq, name=name))
        self.__undefined.remove(name)

    def transit(self, states_and_actions):
        assert len(self.__undefined) == 0, "undefined variables!"

        self.__scm.assign(**states_and_actions)
        outs = {name: self.__scm[name].value for name in self.names_outputs}
        return outs, self.__scm.valuedic()
