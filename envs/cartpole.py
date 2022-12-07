import gym.envs.classic_control.cartpole as cartpole
import numpy as np
from core import Env
from core import ContinuousNormal, TruncatedNormal, \
    Binary, NamedCategorical, Boolean, Categorical
from utils.typings import NamedValues


POSITION = 'position'
VELOCITY = 'velocity'
ANGLE = 'angle'
ANGLE_VLOCITY = 'angle_velocity'
PUSH = 'PUSH'


NEXT = {name: Env.name_next(name)
        for name in (POSITION, VELOCITY, ANGLE, ANGLE_VLOCITY)}


class Cartpole(Env):
    def __init__(self, render=False):
        if render:
            self.__core = cartpole.CartPoleEnv(render_mode='human')
        else:
            self.__core = cartpole.CartPoleEnv()
        
        _def = Env.Definition()
        _def.state(POSITION, ContinuousNormal(scale=None))
        _def.state(VELOCITY, ContinuousNormal(scale=None))
        _def.state(ANGLE, ContinuousNormal(scale=None))
        _def.state(ANGLE_VLOCITY, ContinuousNormal(scale=None))
        _def.action(PUSH, NamedCategorical('left', 'right'))

        super().__init__(_def)

        self.def_reward('failure', [POSITION, ANGLE],
                        lambda x, a: -10 * float(self.__fail(x, a)))
    
    def __state_variables(self, obs):
        x, vx, a, va = obs
        return {POSITION: x, VELOCITY: vx, ANGLE: a, ANGLE_VLOCITY: va}
    
    def __fail(self, pos, ang):
        return np.abs(ang) > self.__core.theta_threshold_radians or \
            np.abs(pos) > self.__core.x_threshold

    def init_episode(self, *args, **kargs):
        obs, info = self.__core.reset()
        return self.__state_variables(obs)
    
    def transit(self, actions):
        a = int(actions[PUSH])

        observation, reward, terminated, truncated, info = \
            self.__core.step(a)  # type: ignore

        s = self.__state_variables(observation)
        tran = {self.name_next(k): s[k] for k in s.keys()}

        return tran, info

    def terminated(self, transition) -> bool:
        return self.__fail(transition[POSITION], transition[ANGLE])

    def random_action(self):
        a = self.__core.action_space.sample()
        return {PUSH: a}
