from typing import Callable, Sequence, Tuple

from .scm import ExoVar, EndoVar, StructrualCausalModel
from .env import EnvInfo, Env, _KArrays


class CausalMdp(Env):

    def __init__(self, envinfo: EnvInfo):
        super().__init__(envinfo)
        self.model = StructrualCausalModel()

        self.__undefined = set(self.names_outputs)
        for name in self.names_inputs:
            self.model.add(ExoVar(name, self.var(name).default))
        
    def define(self, name: str, parents: Sequence[str], eq: Callable):
        if name in self.names_s:
            name = self.name_next_step(name)
        if name not in self.__undefined or name not in self.names_next_s:
            raise ValueError(f"{name} is not a endogenous variable of the environment"
                             " or has already been defined.")
        self.model.add(EndoVar([self.model[pa] for pa in parents], eq, name=name))
        self.__undefined.remove(name)

    def transit(self, states_and_actions):
        assert len(self.__undefined) == 0, "undefined variables!"

        self.model.assign(**states_and_actions)
        outs = {name: self.model[name].value for name in self.names_outputs}
        return outs, self.model.valuedic()
