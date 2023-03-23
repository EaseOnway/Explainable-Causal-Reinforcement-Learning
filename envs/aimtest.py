import numpy as np
from core import Env
from core import ContinuousNormal, TruncatedNormal, \
    Binary, NamedCategorical, Boolean, Categorical
from utils.typings import NamedValues




X1 = 'x1'
X2 = 'x2'
X3 = 'x3'
X4 = 'x4'
A = 'a'
TIME = 't'

NEXT = {name: Env.name_next(name) for name in (X1, X2, X3, X4, TIME)}

class AimTestEnv(Env):

    N_ACTION = 4

    @classmethod
    def init_parser(cls, parser):
        pass

    def define(self, args):
        
        _def = Env.Definition()
        _def.state(X1, ContinuousNormal(scale=None))
        _def.state(X2, ContinuousNormal(scale=None))
        _def.state(X3, ContinuousNormal(scale=None))
        _def.state(X4, ContinuousNormal(scale=None))
        _def.state(TIME, ContinuousNormal(scale=None))
        _def.action(A, Categorical(self.N_ACTION))

        return _def
    
    def init_episode(self, *args, **kargs):
        self.x1 = np.random.normal(10, 1)
        self.x2 = np.random.normal(20, 1)
        self.x3 = 0.
        self.x4 = np.random.normal(15, 1)
        self.time = 0.
        return self.__observe()
    
    def launch(self):
        pass
    
    def __observe(self, next=False):
        d = {
            X1: self.x1,
            X2: self.x2,
            X3: self.x3,
            X4: self.x4,
            TIME: self.time,
        }
        if next:
            return {NEXT[k]: v for k, v in d.items()}
        else:
            return d

    def transit(self, actions):
        a = actions[A]
        x1 = self.x1 + np.random.normal(1, 1)
        x2 = (self.x1 if a==0 else self.x2) + np.random.normal(scale=1)
        x3 = self.x3 + [self.x1, self.x2, 5, 10][a] + np.random.normal(scale=1)
        x4 = 0.1 * self.x3 +  0.9 * self.x4 + np.random.normal(scale=0.5)
        time = self.time + [10, 20, 5, 5][a]

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.time = time

        return self.__observe(next=True), {}

    def terminated(self, transition) -> bool:
        return transition[NEXT[TIME]] >= 1000

    def random_action(self):
        a = np.random.randint(self.N_ACTION)
        return {A: a}

    def action_influence_graph(self, a: int):
        g = {
            NEXT[X1]: (X1,),
            NEXT[X2]: [(X1), (X2,), (X2,), (X2),][a],
            NEXT[X3]: [(X1, X3), (X2, X3), (X3,), (X3,)][a],
            NEXT[X4]: (X3, X4),
            NEXT[TIME]: (TIME,)
        }
        return g

    def structural_causal_graph(self):
        g = {
            NEXT[X1]: (X1,),
            NEXT[X2]: (X1, X2, A),
            NEXT[X3]: (X1, X2, X3, A),
            NEXT[X4]: (X3, X4),
            NEXT[TIME]: (TIME, A)
        }
        return g
