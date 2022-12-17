import gym.envs.toy_text.taxi as taxi
import numpy as np
from core import Env
from core import ContinuousNormal, TruncatedNormal, \
    Binary, NamedCategorical, Boolean, Categorical
from utils.typings import NamedValues


X = 'x'
Y = 'y'
PASSENGER_LOCATION = 'passenger_location'
DESTINATION = 'destination'
ACTION = 'action'
ILLEGAL_OPERATION = 'illegal_operation'
NEXT = {name: Env.name_next(name)
        for name in (X, Y, PASSENGER_LOCATION, DESTINATION)}


class Taxi(Env):

    @classmethod
    def init_parser(cls, parser):
        parser.add_argument("--render", action="store_true", default=False,
                            help="render the envrionment in pygame window.")

    def define(self, args):
        self.__render: bool = args.render
        
        _def = Env.Definition()
        _def.state(X, Categorical(5))
        _def.state(Y, Categorical(5))
        _def.state(PASSENGER_LOCATION, NamedCategorical('red', 'green', 'yellow', 'blue', 'in_taxi'))
        _def.state(DESTINATION, NamedCategorical('red', 'green', 'yellow', 'blue'))
        _def.action(ACTION, NamedCategorical('south', 'north', 'east', 'west', 'pike_up', 'drop'))
        _def.outcome(ILLEGAL_OPERATION, Boolean(scale=None))

        def _r_passenger_picked_up(p, p_):
            if p != 4 and p_ == 4:
                return 10
            else:
                return 0

        _def.reward('passenger_picked_up', (PASSENGER_LOCATION, NEXT[PASSENGER_LOCATION]),
            _r_passenger_picked_up)
        _def.reward('passenger_delivered', (PASSENGER_LOCATION, DESTINATION),
            lambda p, d: 10 if p==d else 0)
        _def.reward('invalid_operation', (ILLEGAL_OPERATION,),
            lambda x: -10 if x else 0)
        
        return _def

    def launch(self):
        if self.__render:
            self.__core = taxi.TaxiEnv(render_mode='human')
        else:
            self.__core = taxi.TaxiEnv()

    def __state_variables(self, obs):
        y, x, p, d = tuple(self.__core.decode(obs))
        return {Y: y, X: x, PASSENGER_LOCATION: p, DESTINATION: d}

    def init_episode(self, *args, **kargs):
        obs, info = self.__core.reset()
        y, x, p, d = tuple(self.__core.decode(obs))
        return {Y: y, X: x, PASSENGER_LOCATION: p, DESTINATION: d}
    
    def transit(self, actions):
        a = int(actions[ACTION])

        observation, reward, terminated, truncated, info = \
            self.__core.step(a)  # type: ignore

        s = self.__state_variables(observation)
        tran = {self.name_next(k): s[k] for k in s.keys()}
        tran[ILLEGAL_OPERATION] = (reward == -10)

        return tran, info

    def terminated(self, transition) -> bool:
        return transition[NEXT[PASSENGER_LOCATION]] == transition[NEXT[DESTINATION]]

    def random_action(self):
        a = self.__core.action_space.sample()
        return {ACTION: a}
